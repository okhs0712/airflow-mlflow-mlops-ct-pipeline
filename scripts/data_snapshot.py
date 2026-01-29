import os
import yaml
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. 설정 로드
    config = load_config("configs/run.yaml")
    seed = config['seed']
    data_config = config['data']
    
    # 2. 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[Info] Random Seed set to {seed}")

    # 3. 데이터 저장 경로 설정
    raw_dir = data_config['raw_dir']
    os.makedirs(raw_dir, exist_ok=True)
    
    # 4. FashionMNIST 다운로드 (Raw Data)
    print("[Info] Downloading FashionMNIST")
    # 전처리는 학습 단계에서 하므로 여기서는 ToTensor만 적용
    full_train_dataset = datasets.FashionMNIST(
        root=raw_dir, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
        root=raw_dir, 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )

    # 5. Train / Val 분할 (Seed=42 적용)
    # configs/run.yaml의 비율 사용 (예: 0.9)
    split_ratio = data_config['split_ratio']
    train_size = int(len(full_train_dataset) * split_ratio)
    val_size = len(full_train_dataset) - train_size
    
    # random_split은 torch.manual_seed의 영향을 받음
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    print(f"[Info] Data Split Completed:")
    print(f"       - Train: {len(train_subset)}")
    print(f"       - Val  : {len(val_subset)}")
    print(f"       - Test : {len(test_dataset)}")

    # 6. 스냅샷 저장 (데이터셋 태그)
    # artifacts/dataset_tag.txt에 태그 저장
    tag = data_config['tag_format'].format(seed)
    os.makedirs("artifacts", exist_ok=True)
    
    with open("artifacts/dataset_tag.txt", "w") as f:
        f.write(tag)
    
    print(f"[Info] Dataset tag saved: {tag} -> artifacts/dataset_tag.txt")

if __name__ == "__main__":
    main()