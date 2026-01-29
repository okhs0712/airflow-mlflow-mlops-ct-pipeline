import click
import subprocess
import sys
import os

# 스크립트 실행 도우미 함수
def run_command(command):
    """쉘 명령어를 실행하고 에러가 나면 종료하는 함수"""
    try:
        # 실시간 로그 출력을 위해 check=True 사용
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"[Error] Command failed: {command}", err=True)
        sys.exit(e.returncode)

@click.group()
def cli():
    """FashionMNIST MLOps Pipeline CLI"""
    pass

# 1. 데이터 스냅샷
@cli.command()
def snapshot():
    """Download and split data with fixed seed."""
    click.echo(">>> [Step 1] Data Snapshot...")
    run_command("python scripts/data_snapshot.py")

# 2. 모델 학습
@cli.command()
@click.option('--model', type=click.Choice(['baseline', 'resnet34', 'efficientnet_b0']), required=True, help='Model to train')
def train(model):
    """Train a specific model."""
    click.echo(f">>> [Step 2] Training Model: {model}...")
    run_command(f"python scripts/train.py --model {model}")

# 3. 모델 평가
@cli.command()
@click.option('--model', type=click.Choice(['baseline', 'resnet34', 'efficientnet_b0']), required=True, help='Model to evaluate')
def evaluate(model):
    """Evaluate a trained model."""
    click.echo(f">>> [Step 3] Evaluating Model: {model}...")
    run_command(f"python scripts/eval.py --model {model}")

# 4. 릴리즈 게이트 (실패해도 파이프라인이 죽으면 안 됨)
@cli.command()
@click.option('--model', required=True, help='Model to check')
def gate(model):
    """Check release gate conditions."""
    click.echo(f">>> [Step 4] Release Gate Check: {model}...")
    # release_gate.py는 실패 시 'FAIL'을 출력하지만, 종료 코드(exit code)는 0(정상)이어야 함.
    # 만약 0이 아니면 여기서 에러가 나서 파이프라인이 멈춤. 
    run_command(f"python tools/release_gate.py --model {model}")

# 5. 레지스트리 등록
@cli.command()
@click.option('--model', required=True, help='Model to register')
def register(model):
    """Register passed model to registry & MLflow."""
    click.echo(f">>> [Step 5] Registering Model: {model}...")
    run_command(f"python tools/register_mlflow.py --model {model}")

if __name__ == '__main__':
    cli()