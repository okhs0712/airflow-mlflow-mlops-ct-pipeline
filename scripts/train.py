import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import click
import time

# 설정 파일 로드
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 1. 모델 공장 (Model Factory): 이름만 주면 모델을 만들어주는 함수
def get_model(model_name, config):
    # 클래스 개수 (FashionMNIST = 10개)
    num_classes = 10
    
    if model_name == "baseline":
        # Baseline: 학습되지 않은 간단한 CNN 
        # 입력이 224x224, 3채널로 들어오므로 그에 맞춰 설계
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, num_classes) 
        )
    
    elif model_name == "resnet34":
        # ResNet34: ImageNet Pretrained 
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # 마지막 레이어를 우리 데이터(10개 클래스)에 맞게 교체
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        # EfficientNet_B0: ImageNet Pretrained 
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 마지막 레이어 교체
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

@click.command()
@click.option('--model', type=click.Choice(['baseline', 'resnet34', 'efficientnet_b0']), required=True, help='학습할 모델 선택')
def train(model):
    # 설정 로드
    config = load_config("configs/run.yaml")
    seed = config['seed']
    
    # 시드 설정
    torch.manual_seed(seed)
    
    # CPU 스레드 수 설정
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    
    # 2. 전처리 정의 
    # MNIST(1채널) -> 3채널 복사 -> 224x224 리사이즈 -> 정규화
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 1채널 -> 3채널
        transforms.Resize((config['train']['image_size'], config['train']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 데이터 로드 (Data Snapshot 단계와 동일한 시드로 분할)
    raw_dir = config['data']['raw_dir']
    full_dataset = datasets.FashionMNIST(root=raw_dir, train=True, download=True, transform=transform)
    
    split_ratio = config['data']['split_ratio']
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size
    
    train_subset, _ = random_split(full_dataset, [train_size, val_size])
    
    # 데이터 로더 생성
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=4, persistent_workers=True)
    
    # 4. 모델 및 학습 설정
    device = torch.device("cpu") # 과제 요구사항: CPU 제한 
    net = get_model(model, config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['models'][model]['learning_rate'])
    epochs = config['models'][model]['epochs']
    
    print(f"[{model}] Training Start... (Epochs: {epochs}, Device: {device})")
    
    # 5. 학습 루프
    net.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 로그 출력 (100배치마다)
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
    print(f"[{model}] Training Finished. Time: {time.time() - start_time:.2f}s")
    
    # 6. 모델 저장 
    save_dir = f"artifacts/{model}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/model.pt"
    
    torch.save(net.state_dict(), save_path)
    print(f"[{model}] Model saved to {save_path}")

if __name__ == '__main__':
    train()