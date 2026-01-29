import os
import json
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import click
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize

# 설정 로드
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 모델 정의 (train.py와 동일)
def get_model(model_name, num_classes=10):
    if model_name == "baseline":
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, num_classes)
        )
    elif model_name == "resnet34":
        model = models.resnet34(weights=None) # 평가는 학습된 가중치(pt)를 로드하므로 weights=None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# 1. 지연 시간(Latency) 측정 함수 
def measure_latency(model, device, image_size=224):
    model.eval()

    # 스레드 수 제한 (CPU 평가용)
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    latencies = []
    
    # 더미 입력 (배치=1, 3채널, 224x224)
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # 워밍업 (10회 제외)
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
            
        # 실제 측정 (200회)
        for _ in range(200):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # ms 단위 변환

    # p95 계산
    p95_latency = np.percentile(latencies, 95)
    return p95_latency

# 2. 성능 평가 함수 (Acc, F1)
def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') # Multi-class Avg F1
    
    return acc, f1, all_labels, all_preds

# 3. Youden Index 기반 최적 Cutoff 계산
def calculate_optimal_cutoff(model, dataloader, device, num_classes=10):
    model.eval()
    all_probs = []
    all_labels = []

    # 1. Validation Set에 대한 확률값(Probability) 수집
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Softmax로 확률 변환 필수
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 2. 라벨 이진화 (One-vs-Rest 방식을 위해)
    # 예: 0번 클래스 vs 나머지, 1번 클래스 vs 나머지...
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    
    optimal_thresholds = []

    # 3. 각 클래스별 Youden Index 계산
    for i in range(num_classes):
        # i번째 클래스에 대한 ROC Curve 계산
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], all_probs[:, i])
        
        # Youden Index = Sensitivity(TPR) + Specificity(1-FPR) - 1
        # J = TPR - FPR
        J = tpr - fpr
        
        # J가 최대가 되는 인덱스 찾기
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        optimal_thresholds.append(best_thresh)
        
    # 4. 전체 클래스의 평균 Cutoff 반환 (대표값)
    # 실제로는 클래스별로 다르게 적용해야 하지만, JSON 기록용으로는 평균이 적합
    avg_cutoff = np.mean(optimal_thresholds)
    
    print(f"   -> [Youden Index] Class-wise optimal thresholds calculated.")
    print(f"   -> Avg Optimal Cutoff: {avg_cutoff:.4f}")
    
    return avg_cutoff

@click.command()
@click.option('--model', type=click.Choice(['baseline', 'resnet34', 'efficientnet_b0']), required=True, help='평가할 모델')
def evaluate(model):
    config = load_config("configs/run.yaml")
    seed = config['seed'] # 42 사용
    torch.manual_seed(seed)
    
    device = torch.device("cpu") # 평가도 CPU 
    print(f"[{model}] Evaluation Start on {device}...")

    # 1. 모델 로드
    net = get_model(model).to(device)
    model_path = f"artifacts/{model}/model.pt"
    
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        return

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 2. 데이터셋 준비
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_dir = config['data']['raw_dir']
    full_dataset = datasets.FashionMNIST(root=raw_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=raw_dir, train=False, download=True, transform=transform)
    
    # Val Set 분리 (최적 Cutoff 계산용)
    split_ratio = config['data']['split_ratio']
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size

    # 고정 시드 사용    
    g = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(full_dataset, [train_size, val_size], generator=g)

    val_loader = DataLoader(val_subset, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 3. Val Set에서 Youden Index로 Optimal Cutoff 선택
    print(f"[{model}] calculating Optimal Cutoff (Youden Index) on Val Set...")
    optimal_cutoff = calculate_optimal_cutoff(net, val_loader, device)

    # 4. Latency & Performance 측정 (Test Set)
    print(f"[{model}] Measuring Performance on Test Set...")
    p95_ms = measure_latency(net, device)
    acc, f1, _, _ = evaluate_metrics(net, test_loader, device)

    print(f"   -> Top-1 Acc: {acc:.4f}, Avg F1: {f1:.4f}, p95: {p95_ms:.2f}ms")
    
    # 5. 결과 저장
    eval_result = {
        "model_name": model,
        "acc": round(acc, 4),
        "f1_score": round(f1, 4),
        "p95_latency_ms": round(p95_ms, 2),
        "cutoff_threshold": round(float(optimal_cutoff), 4), # 계산된 값 저장
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_path = f"artifacts/{model}/eval.json"
    with open(output_path, "w") as f:
        json.dump(eval_result, f, indent=4)
        
    print(f"[{model}] Evaluation Saved: {output_path}")

if __name__ == '__main__':
    evaluate()