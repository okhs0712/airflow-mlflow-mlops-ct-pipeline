import os
import json
import yaml
import sys
import click

# 설정 파일 로드
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.command()
@click.option('--model', required=True, help='평가할 모델 이름')
def check_gate(model):
    # 1. 설정 및 평가 결과 로드
    config = load_config("configs/run.yaml")
    eval_path = f"artifacts/{model}/eval.json"
    
    # 평가 파일이 없으면 FAIL 처리
    if not os.path.exists(eval_path):
        print("FAIL")
        print(f"[FAIL] {model}: Evaluation result not found at {eval_path}")
        return

    with open(eval_path, 'r') as f:
        results = json.load(f)

    # 2. 임계값 가져오기
    thresholds = config['evaluation']['thresholds']
    min_acc = thresholds['min_accuracy']
    min_f1 = thresholds['min_f1_score']
    max_latency = thresholds['max_latency']

    # 3. 평가 수행
    acc = results.get('acc', 0.0)
    f1 = results.get('f1_score', 0.0)
    latency = results.get('p95_latency_ms', 9999.0)
    
    reasons = []
    is_pass = True

    # 조건 1: 정확도
    if acc < min_acc:
        is_pass = False
        reasons.append(f"Low Accuracy ({acc:.4f} < {min_acc})")
    
    # 조건 2: F1 Score
    if f1 < min_f1:
        is_pass = False
        reasons.append(f"Low F1 Score ({f1:.4f} < {min_f1})")
        
    # 조건 3: 지연 시간
    if latency > max_latency:
        is_pass = False
        reasons.append(f"High Latency ({latency:.2f}ms > {max_latency}ms)")

    # 4. 결과 출력
    if is_pass:
        print("PASS")
        print(f"[Info] {model} passed release gate.")
    else:
        print("FAIL")
        print(f"[Info] {model} failed release gate. Reasons: {', '.join(reasons)}")

if __name__ == '__main__':
    check_gate()