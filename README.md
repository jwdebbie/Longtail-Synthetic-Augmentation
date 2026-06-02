# longtail-synthetic-augmentation

> **Synthetic Data Augmentation Experiments for Long-Tail Image Classification**

Long-Tail 환경에서 Synthetic data 보완이 Tail class 성능에 미치는 영향을 검증하는 실험 파이프라인입니다.  
CIFAR-100 LT와 CIFAR-10 LT 두 가지 설정에서 실험을 진행했습니다.

---

## Overview

현실의 데이터는 균등하지 않습니다. 특정 class는 데이터가 넘치고, 희귀한 class는 극단적으로 부족한 Long-Tail 분포가 일반적입니다.

이 실험은 두 가지 질문에 답합니다.

- **Synthetic data를 Tail class에 보완하면 성능이 오르는가?**
- **생성 모델의 도메인 일치 여부가 성능에 얼마나 영향을 미치는가?**

---

## Results

### CIFAR-10 LT (IF=100) — EDM 생성 모델

CIFAR-10으로 직접 학습된 class-conditional 생성 모델 EDM 사용.

| 실험 | Overall | Head | Tail |
|---|---|---|---|
| Real only | 71~73% | 88~89% | 55~56% |
| +EDM Synthetic 25% | 89~90% | 91~92% | 87~88% |

> EDM이 CIFAR-10 도메인으로 직접 학습된 모델이라 FID가 낮고 Real data 분포와 유사해 Tail 성능이 크게 향상됨.

### CIFAR-100 LT (IF=100) — SDXL-Turbo 생성 모델

| 실험 | Overall | Head | Tail |
|---|---|---|---|
| Real only | 19.33% | 31.96% | 6.70% |
| +Synthetic | 24.15% | 38.78% | 9.52% |

---

## Dataset

**Exponential Decay 방식으로 Long-Tail 분포 구성**

```
class i 샘플 수 = max_samples × (1/IF)^(i / (n_classes - 1))
```

| IF 설정 | 난이도 | Head | Tail |
|---|---|---|---|
| IF = 10 | Easy | 5,000개 | 500개 |
| IF = 50 | Medium | 5,000개 | 100개 |
| IF = 100 | Hard | 5,000개 | 50개 |

**CIFAR-10 LT (IF=100)**
```
class 0 (plane):  5,000개  ← Head
class 1 (car):    2,997개
...
class 9 (truck):     50개  ← Tail
총 train: 12,408개 (원본 50,000개)
```

---

## File Structure

```
longtail-synthetic-augmentation/
├── cifar10/
│   ├── main.py                     # CIFAR-10 LT 학습 (kuangliu 기반)
│   └── train_cifar10_lt.py         # CIFAR-10 LT + Synthetic fine-tuning
├── cifar100/
│   ├── make_longtail.py            # Long-Tail 데이터셋 생성
│   ├── generate_cifar_synthetic.py # SDXL-Turbo Synthetic 생성
│   └── train_cifar_lt.py           # CIFAR-100 LT + Synthetic 학습
├── requirements.txt
└── README.md
```

---


## Requirements

```bash
pip install torch torchvision
pip install datasets diffusers transformers
pip install pillow numpy
```

---

## References

- Cao et al. *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (NeurIPS 2019)
- Fan et al. *Scaling Laws of Synthetic Images for Model Training* (2024)
- Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models* (NeurIPS 2022)