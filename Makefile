# Makefile
# 목적:
# - make run 한 번으로 Airflow 기반 CT 파이프라인을 실행 시작
# - (snapshot → train → eval → gate → register)
# - 실행은 Airflow가 백그라운드에서 수행

PUBLIC_IP := $(shell curl -s ifconfig.me)
export PUBLIC_IP

# Docker Compose 설정 파일
COMPOSE_FILE = Docker-compose.airflow.yml

# Airflow 컨테이너 이름
CONTAINER_NAME = airflow_standalone

# 실행할 Airflow DAG ID
DAG_ID = fashion_mnist_pipeline

# Airflow Web UI 주소
AIRFLOW_BASE_URL = http://localhost:8080

.PHONY: run up down trigger logs wait

# 원클릭 실행
# 1) Airflow 환경 실행
# 2) Webserver 준비 대기
# 3) DAG 트리거
run: up wait trigger

# Airflow 컨테이너 실행
up:
	PUBLIC_IP=$$(curl -s ifconfig.me) docker compose -f $(COMPOSE_FILE) up -d

# Airflow 컨테이너 종료
down:
	docker compose -f $(COMPOSE_FILE) down

# Airflow Webserver 준비 상태 확인
# /health 엔드포인트가 응답할 때까지 대기
wait:
	@echo "Waiting for Airflow webserver..."
	@for i in $$(seq 1 120); do \
	  code=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v2/version || true); \
	  if [ "$$code" = "200" ]; then \
	    echo "Airflow is ready."; \
	    exit 0; \
	  fi; \
	  sleep 2; \
	done; \
	echo "Airflow is not ready (timeout)."; \
	exit 1

# DAG 실행
# - 이미 unpause 상태여도 에러 없이 진행
trigger:
	docker exec $(CONTAINER_NAME) airflow dags unpause $(DAG_ID) >/dev/null 2>&1 || true
	docker exec $(CONTAINER_NAME) airflow dags trigger $(DAG_ID)
	@echo "Triggered DAG: $(DAG_ID)"
	@echo "UI: $(AIRFLOW_BASE_URL)/dags/$(DAG_ID)/grid"

# Airflow 컨테이너 로그 확인
logs:
	docker logs -f $(CONTAINER_NAME)
