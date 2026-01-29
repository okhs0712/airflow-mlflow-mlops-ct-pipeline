import os
import json
import yaml
import time
import datetime
import click
import torch
import torch.nn as nn
from torchvision import models
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# 모델을 등록하기 위해 train.py의 모델 공장(Factory) 로직을 그대로 가져옴
def create_model_architecture(model_name):
    num_classes = 10
    
    if model_name == "baseline":
        # train.py와 100% 동일한 구조여야 함
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, num_classes) 
        )
    
    elif model_name == "resnet34":
        # Pretrained=False (구조만 필요하므로 가중치는 나중에 덮어씌움)
        model = models.resnet34(weights=None) 
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.command()
@click.option('--model', required=True, help='등록할 모델 이름')
def register(model):
    print(f"🚀 [Start] Registering process for model: {model}")

    # 1. 설정 및 평가 결과 로드
    config = load_config("configs/run.yaml")
    
    eval_path = f"artifacts/{model}/eval.json"
    if not os.path.exists(eval_path):
        print(f"[Error] No evaluation result found for {model}")
        return

    with open(eval_path, 'r') as f:
        eval_result = json.load(f)
        
    dataset_tag = "unknown"
    if os.path.exists("artifacts/dataset_tag.txt"):
        with open("artifacts/dataset_tag.txt", "r") as f:
            dataset_tag = f.read().strip()

    # 2. 메타데이터 생성 (JSONL 저장용)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"mnist_{model}_{timestamp}"
    
    meta_data = {
        "model_id": model_id,
        "dataset_tag": dataset_tag,
        "metrics": {
            "acc": eval_result.get('acc'),
            "f1": eval_result.get('f1_score'),
            "p95_ms": eval_result.get('p95_latency_ms')
        },
        "artifacts": {
            "ckpt": f"artifacts/{model}/model.pt",
            "eval_json": eval_path
        },
        "stage": "staging", # 테스트 통과했으므로 staging
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 3. JSONL 파일에 저장 (Minimum Requirement)
    os.makedirs("registry", exist_ok=True)
    registry_path = "registry/metadata.jsonl"
    
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta_data) + "\n")
        
    print(f"✅ [Success] Metadata saved to {registry_path}")

    # 4. MLflow Model Registry 등록
    print("-------------------------------------------------------")
    print("🌊 Attempting MLflow Model Registration...")

    try:
        # 환경변수 사용, 없으면 localhost
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("fashion_mnist_ct")
        print(f"   -> MLflow URI: {uri}")

        # 모델 파일 경로 확인
        model_file_path = f"artifacts/{model}/model.pt"
        if not os.path.exists(model_file_path):
            print(f"[Error] Model file not found: {model_file_path}")
            return

        # 1) 빈 모델 생성
        print(f"   -> Reconstructing model architecture: {model}...")
        model_obj = create_model_architecture(model)

        # 2) 저장된 가중치(State Dict) 로드
        # CPU 환경에서도 돌아가도록 map_location 설정
        device = torch.device("cpu")
        state_dict = torch.load(model_file_path, map_location=device)

        # 3) 가중치 주입
        print("   -> Loading state_dict into the model...")
        model_obj.load_state_dict(state_dict)
        print("   -> ✨ Model successfully reconstructed!")

        # 4) MLflow에 모델 등록
        with mlflow.start_run(run_name=model_id) as run:
            # 파라미터 및 메트릭 기록
            mlflow.log_param("model_type", model)
            mlflow.log_param("dataset_tag", dataset_tag)
            mlflow.log_metrics(meta_data["metrics"])
            
            # 평가 결과 파일 업로드
            mlflow.log_artifact(eval_path, artifact_path="eval_results")

            # log_model 사용
            mlflow.pytorch.log_model(model_obj, name="model")
            
            # 레지스트리 등록
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, f"fashion_mnist_{model}")
            
            # 모델 설명 추가
            client = MlflowClient()
            client.update_model_version(
                name=f"fashion_mnist_{model}",
                version=registered_model.version,
                description=f"Auto-registered via Pipeline. Acc: {meta_data['metrics']['acc']}"
            )
            
            # Staging으로 승격
            # 2.9.0부터 deprecated
            client.transition_model_version_stage(
                name=f"fashion_mnist_{model}",
                version=registered_model.version,
                stage="Staging"
            )
            
        print(f"🌟 [Bonus Complete] Registered to MLflow: fashion_mnist_{model} (v{registered_model.version})")
        
    except Exception as e:
        # 실패해도 파이프라인은 죽이지 않음
        print(f"⚠️ [Bonus Failed] MLflow registration error: {e}")
        print("   -> Minimum requirement (JSONL) is safe. Pipeline continues.")

if __name__ == '__main__':
    register()