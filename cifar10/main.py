'''Train CIFAR10 with PyTorch.'''

import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, ConcatDataset
from PIL import Image
from pathlib import Path

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--longtail', action='store_true',
                    help='use long tail dataset')
parser.add_argument('--longtail_stats', type=str,
                    default='../dataset/cifar10_longtail/longtail_stats.json',
                    help='path to longtail_stats.json')
# Synthetic 관련 옵션
parser.add_argument('--synthetic', action='store_true',
                    help='add synthetic data')
parser.add_argument('--synth_data_root', type=str,
                    default='../dataset/synth_cifar10_edm',
                    help='path to EDM synthetic data')
parser.add_argument('--fill_ratio', type=float, default=0.25,
                    help='부족분의 몇 %를 synthetic으로 채울지 (0.25/0.5/0.75/1.0)')
parser.add_argument('--checkpoint_path', type=str,
                    default='./checkpoint/ckpt_lt_real.pth',
                    help='Real only checkpoint 경로 (synthetic 실험 시 사용)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# 전처리 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Synthetic Dataset 
class SyntheticDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# LT Subset 
class LTSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, torch.tensor(label, dtype=torch.long)


# 데이터 로드
print('==> Preparing data..')
full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

label_counts = {}

if args.longtail:
    with open(args.longtail_stats) as f:
        stats = json.load(f)

    random.seed(42)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_trainset):
        label_to_indices[label].append(idx)

    selected = []
    for cls_idx, info in stats['class_distribution'].items():
        n = info['n_samples']
        indices = label_to_indices[int(cls_idx)]
        sampled = random.sample(indices, min(n, len(indices)))
        selected.extend(sampled)
        label_counts[int(cls_idx)] = len(sampled)

    trainset = LTSubset(full_trainset, selected)
    print(f'Long Tail Train samples: {len(trainset)}')
    for i in range(10):
        print(f'  class {i} ({CIFAR10_CLASSES[i]}): {label_counts[i]}개')
else:
    trainset = full_trainset
    print(f'Standard Train samples: {len(trainset)}')

# Synthetic 추가 
if args.synthetic and args.longtail:
    max_per_class = 5000
    root = Path(args.synth_data_root)
    synth_samples = []

    print(f'\n==> Adding synthetic data (fill_ratio={args.fill_ratio:.0%})')
    for cls_idx in range(10):
        real_count = label_counts.get(cls_idx, 0)
        deficit = max(0, max_per_class - real_count)
        n_to_add = int(deficit * args.fill_ratio)

        if n_to_add == 0:
            continue

        class_dir = root / f"class{cls_idx}"
        if not class_dir.exists():
            print(f'  class {cls_idx} synthetic 폴더 없음')
            continue

        all_imgs = sorted(class_dir.glob("*.png"))
        if not all_imgs:
            all_imgs = sorted(class_dir.glob("*.jpg"))

        if len(all_imgs) >= n_to_add:
            chosen = random.sample(all_imgs, n_to_add)
        else:
            chosen = random.choices(all_imgs, k=n_to_add)

        synth_samples.extend([(str(p), cls_idx) for p in chosen])
        print(f'  class {cls_idx} ({CIFAR10_CLASSES[cls_idx]}): '
              f'real {real_count} + synthetic {len(chosen)} '
              f'(deficit {deficit}의 {args.fill_ratio:.0%})')

    synth_ds = SyntheticDataset(synth_samples, transform=transform_train)
    trainset = ConcatDataset([trainset, synth_ds])
    print(f'Combined: {len(trainset)}개 (synthetic {len(synth_samples)}개 추가)')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

# Test set 
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = CIFAR10_CLASSES

# 모델 
print('==> Building model..')
net = ResNet50()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Synthetic 실험 시 Real only checkpoint 불러오기
if args.synthetic:
    print(f'==> Loading Real only checkpoint: {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    print(f'==> Loaded! (acc: {checkpoint["acc"]:.2f}%)')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for t, p in zip(targets, predicted):
                class_total[t.item()] += 1
                if t == p:
                    class_correct[t.item()] += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

    if args.longtail:
        with open(args.longtail_stats) as f:
            stats = json.load(f)
        counts = [stats['class_distribution'][str(i)]['n_samples'] for i in range(10)]
        median = sorted(counts)[len(counts) // 2]

        head_correct = sum(class_correct[i] for i in range(10) if counts[i] >= median)
        head_total = sum(class_total[i] for i in range(10) if counts[i] >= median)
        tail_correct = sum(class_correct[i] for i in range(10) if counts[i] < median)
        tail_total = sum(class_total[i] for i in range(10) if counts[i] < median)

        head_acc = 100. * head_correct / head_total if head_total > 0 else 0
        tail_acc = 100. * tail_correct / tail_total if tail_total > 0 else 0
        print(f'\nOverall: {acc:.2f}% | Head: {head_acc:.2f}% | Tail: {tail_acc:.2f}%')

    # checkpoint 저장 (실험별로 다른 파일명)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.synthetic:
            ckpt_name = f'ckpt_lt_synth_{int(args.fill_ratio*100)}.pth'
        else:
            ckpt_name = 'ckpt_lt_real.pth'
        torch.save(state, f'./checkpoint/{ckpt_name}')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()