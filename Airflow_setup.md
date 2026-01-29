# Airflow Setup Guide

본 문서는 **Docker Compose 기반으로 구성된 Airflow / MLflow 실행 환경**을 설명합니다.

- 어떤 서비스가 실행되는지
- 왜 해당 설정을 사용했는지
- CT 파이프라인과 어떻게 연결되는지
  를 명확히 기술하는 데 목적을 둡니다.

Airflow DAG의 태스크 구성, 실행 흐름, 분기 설계에 대한 설명은 README.md의 **5. Pipeline Design** 섹션을 참조하시기 바랍니다.

---

## 1. 전체 구성 개요

본 CT 파이프라인은 다음 두 개의 핵심 서비스로 구성됩니다.

- **MLflow Tracking Server**: 실험 메타데이터 및 Model Registry 관리
- **Apache Airflow (Standalone)**: CT 파이프라인 오케스트레이션

두 서비스는 `Docker-compose.airflow.yml`을 통해 동시에 실행되며, 각 서비스의 메타데이터 저장소로 SQLite 기반 DB를 사용하여 로컬 단일 노드 환경에서 **원클릭 실행 및 재현성**을 확보하도록 설계되었습니다.

---

## 2. MLflow Tracking Server 설정

MLflow는 모델 학습 결과, 평가 메트릭, 레지스트리 메타데이터 관리를 위해 사용됩니다.
본 과제에서 MLflow는 모델 파일 자체의 버저닝보다는, 릴리즈 게이트를 통과한 모델에 대해
- 실험 단위의 메트릭 기록
- 모델 식별을 위한 메타데이터 관리
- **PASS 모델의 Model Registry 등록(staging)**
을 목적으로 사용됩니다.

### 사용 이미지

- `ghcr.io/mlflow/mlflow:v3.7.0`

### 주요 설정

- Backend Store: SQLite (`mlflow_db/mlflow.db`)
- Artifact Store: 로컬 디렉토리 공유 (`mlruns/`)
- Serving Port: `5000`

### allowed-hosts 설정

MLflow 3.x부터는 접근 가능한 host를 명시적으로 허용해야 하므로, 다음 host들을 `--allowed-hosts` 옵션에 포함하였습니다.

- 컨테이너 내부 접근 (`mlflow_server`)
- 로컬 접근 (`localhost`, `127.0.0.1`)
- AWS 등에서의 실행을 위한 퍼블릭 IP

퍼블릭 IP는 실행 환경에 따라 달라질 수 있으므로, 환경 변수 `PUBLIC_IP`를 통해 동적으로 주입하도록 구성하였습니다.

### 볼륨 마운트

- `./mlflow_db` → `/mlflow_db`
- `${PWD}/mlruns` → `${PWD}/mlruns`

이를 통해 컨테이너 재시작 이후에도 실험 결과 및 레지스트리 메타데이터가 유지됩니다.

---

## 3. Airflow Standalone 설정

Airflow는 CT 파이프라인 전체를 오케스트레이션하는 역할을 수행합니다.
본 과제에서는 과제의 목적과 실행 환경을 고려하여 Standalone 모드를 사용했습니다.

### 사용 이미지

- `apache/airflow:3.1.5-python3.12`

### 실행 방식

Airflow는 Standalone 모드로 실행되며, 웹서버, 스케줄러, 트리거러가 하나의 컨테이너에서 동작합니다.

컨테이너 시작 시 다음 절차가 수행됩니다.

1. `requirements.txt` 기반 추가 Python 패키지 설치
2. Airflow DB migration 수행
3. Airflow Standalone 실행

### Executor 설정

- Executor: `SequentialExecutor`
- Backend DB: SQLite (`/opt/airflow/db/airflow.db`)


### 환경 변수 설정

- `AIRFLOW__CORE__LOAD_EXAMPLES=False`
- `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=True`

이를 통해 예제 DAG 로딩을 비활성화하고, 로컬 환경에서 별도 사용자 생성 없이 UI 접근이 가능하도록 구성하였습니다.

---

## 4. 볼륨 마운트 및 코드 연동

Airflow 컨테이너에는 CT 파이프라인 실행에 필요한 코드와 결과물이 마운트됩니다.

### 코드 및 설정

- `./DAG` → `/opt/airflow/dags`
- `./scripts` → `/opt/airflow/scripts`
- `./configs` → `/opt/airflow/configs`
- `./tools` → `/opt/airflow/tools`
- `./cli` → `/opt/airflow/cli`

### 결과 및 메타데이터

- `./artifacts` → `/opt/airflow/artifacts`
- `./registry` → `/opt/airflow/registry`
- `./airflow_db` → `/opt/airflow/db`
- `${PWD}/mlruns` → `${PWD}/mlruns`

이러한 마운트 구성을 통해 Airflow 태스크에서 생성된 결과물이 호스트 파일 시스템에 그대로 보존됩니다.

---

## 5. 네트워크 및 실행 의존성

- Airflow Web UI: [http://localhost:8080](http://localhost:8080)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

Airflow 컨테이너는 `depends_on` 설정을 통해 MLflow 서비스가 먼저 실행된 이후 기동되도록 구성되어 있습니다.

---

## 6. 리소스 관련 설정

- `shm_size: 1g`

CPU 기반 학습 및 추론 과정에서 PyTorch DataLoader 및 모델 추론 안정성을 확보하기 위해 공유 메모리 크기를 명시적으로 설정하였습니다.

---

## 7. 정리

본 Docker Compose 기반 Airflow / MLflow 설정은 과제 요구사항을 충족하면서도 원클릭 실행과 로컬 환경 재현성을 확보하는 것을 목표로 설계되었습니다.

Airflow는 파이프라인 제어 및 분기 로직을 담당하고, MLflow는 PASS 모델의 실험 및 레지스트리 메타데이터 관리 역할을 수행합니다.

전체 CT 파이프라인의 동작 흐름과 결과 요약은 README.md를 참고하시기 바랍니다.

