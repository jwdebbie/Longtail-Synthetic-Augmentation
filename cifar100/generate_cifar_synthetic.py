"""
CIFAR-100 Synthetic Data 생성 스크립트 (SDXL-Turbo)

ECCV 2024 Dataset Distillation Challenge 방식 참고:
  - SDXL-Turbo로 CIFAR-100 class label을 텍스트 프롬프트로 사용
  - 1 step sampling으로 빠른 생성
  - class당 N개 이미지 생성
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CIFAR-100 class 이름 목록
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def load_pipeline(device: str = "cuda") -> AutoPipelineForText2Image:
    """SDXL-Turbo 파이프라인 로드."""
    logger.info("Loading SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("SDXL-Turbo loaded!")
    return pipe


def get_tail_classes(longtail_stats_path: str) -> List[int]:
    """longtail_stats.json에서 tail class index 목록 추출."""
    with open(longtail_stats_path) as f:
        stats = json.load(f)

    class_dist = stats["class_distribution"]
    all_counts = [v["n_samples"] for v in class_dist.values()]
    median = sorted(all_counts)[len(all_counts) // 2]

    tail_indices = [
        int(idx)
        for idx, v in class_dist.items()
        if v["n_samples"] < median
    ]
    logger.info("Tail classes: %d개 (median=%d)", len(tail_indices), median)
    return tail_indices


def generate_for_class(
    pipe: AutoPipelineForText2Image,
    class_name: str,
    class_idx: int,
    output_dir: Path,
    n_per_class: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
) -> int:
    """
    하나의 class에 대해 이미지 생성 후 저장.
    Returns: 생성된 이미지 수
    """
    # class 이름 정리 (언더스코어 → 공백)
    class_name_clean = class_name.replace("_", " ")
    prompt = f"a photo of {class_name_clean}"

    # 저장 폴더
    class_dir = output_dir / "train" / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 수 확인 (이어서 생성 가능)
    existing = list(class_dir.glob("*.jpg"))
    start_idx = len(existing)

    if start_idx >= n_per_class:
        logger.info("  [%s] 이미 %d개 존재, 스킵", class_name, start_idx)
        return 0

    remaining = n_per_class - start_idx
    count = 0

    for batch_start in range(0, remaining, batch_size):
        batch_n = min(batch_size, remaining - batch_start)

        with torch.no_grad():
            result = pipe(
                prompt=[prompt] * batch_n,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=image_size,
                width=image_size,
            )

        for i, img in enumerate(result.images):
            # CIFAR-100 해상도(32x32)로 리사이즈
            img_resized = img.resize((image_size, image_size), Image.LANCZOS)
            fname = f"syn_{start_idx + count:06d}.jpg"
            img_resized.save(class_dir / fname, quality=95)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset/synth_cifar",
        help="생성 이미지 저장 경로",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=500,
        help="class당 생성할 이미지 수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="한 번에 생성할 이미지 수",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="SDXL-Turbo는 1 step으로도 충분",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="SDXL-Turbo는 guidance_scale=0 권장",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="생성 이미지 크기 (CIFAR-100: 32)",
    )
    parser.add_argument(
        "--tail_only",
        action="store_true",
        help="tail class만 생성할지 여부",
    )
    parser.add_argument(
        "--longtail_stats",
        type=str,
        default="./dataset/cifar100_longtail/longtail_stats.json",
        help="tail class 판별용 longtail_stats.json 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # tail class 목록
    target_indices = None
    if args.tail_only:
        target_indices = get_tail_classes(args.longtail_stats)
        logger.info("Tail class만 생성: %d개", len(target_indices))
    else:
        target_indices = list(range(len(CIFAR100_CLASSES)))
        logger.info("전체 class 생성: %d개", len(target_indices))

    # 파이프라인 로드
    pipe = load_pipeline(device=args.device)

    # 생성 시작
    total_generated = 0
    output_dir = Path(args.output_dir)

    for class_idx in target_indices:
        class_name = CIFAR100_CLASSES[class_idx]
        logger.info(
            "[%3d/%d] Class: %s",
            target_indices.index(class_idx) + 1,
            len(target_indices),
            class_name,
        )

        n = generate_for_class(
            pipe=pipe,
            class_name=class_name,
            class_idx=class_idx,
            output_dir=output_dir,
            n_per_class=args.n_per_class,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
        )
        total_generated += n
        logger.info("  → %d개 생성 완료", n)

    # 통계 저장
    stats = {
        "model": "stabilityai/sdxl-turbo",
        "n_per_class": args.n_per_class,
        "total_generated": total_generated,
        "tail_only": args.tail_only,
        "target_classes": [CIFAR100_CLASSES[i] for i in target_indices],
        "image_size": args.image_size,
    }
    with open(os.path.join(args.output_dir, "generation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("\n===== 생성 완료 =====")
    logger.info("총 생성: %d개", total_generated)
    logger.info("저장 경로: %s", args.output_dir)


if __name__ == "__main__":
    main()