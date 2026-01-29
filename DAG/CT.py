from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta


MODELS = ["baseline", "resnet34", "efficientnet_b0"]

default_args = {
    "owner": "HyunSeung Jung",
    "depends_on_past": False,
    "start_date": datetime(2025, 12, 20),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

def branch_on_gate_pass(model: str, **context):
    """
    gate task의 XCom(last line)으로 PASS/FAIL을 판단해서
    register 또는 skip으로 분기
    """
    ti = context["ti"]
    # gate task가 TaskGroup 안에 있으므로 task_id는 "{group_id}.gate"
    gate_result = ti.xcom_pull(task_ids=f"{model}.gate")
    gate_result = (gate_result or "").strip()

    if gate_result.startswith("PASS"):
        return f"{model}.register"
    return f"{model}.skip_register"


with DAG(
    dag_id="fashion_mnist_pipeline",
    default_args=default_args,
    description="MLOps Pipeline for FashionMNIST (3 models)",
    schedule="@once",
    catchup=False,
    tags=["mlops"],
) as dag:

    # 1) snapshot (1번만 실행)
    t_snapshot = BashOperator(
        task_id="data_snapshot",
        bash_command="cd /opt/airflow && python /opt/airflow/cli/ct.py snapshot",
        env={"PYTHONUNBUFFERED": "1"},
    )
    prev_end = None
    # 모델별 파이프라인 생성
    for m in MODELS:
        with TaskGroup(group_id=m) as tg:
            # 2) train (실패해도 파이프라인 계속)
            t_train = BashOperator(
                task_id="train",
                bash_command=f"""
                set +e
                cd /opt/airflow
                python /opt/airflow/cli/ct.py train --model {m}
                rc=$?
                if [ $rc -ne 0 ]; then
                  echo "[WARN] train failed ({m}) rc=$rc"
                fi
                exit 0
                """,
                env={"PYTHONUNBUFFERED": "1"},
            )

            # 3) eval (실패해도 계속)
            t_eval = BashOperator(
                task_id="eval",
                bash_command=f"""
                set +e
                cd /opt/airflow
                python /opt/airflow/cli/ct.py evaluate --model {m}
                rc=$?
                if [ $rc -ne 0 ]; then
                  echo "[WARN] eval failed ({m}) rc=$rc"
                fi
                exit 0
                """,
                env={"PYTHONUNBUFFERED": "1"},
            )

            # 4) gate
            # - PASS/FAIL을 마지막 줄로 출력해서 XCom으로 넘김(do_xcom_push=True)
            # - FAIL이어도 exit 0으로 처리해서 DAG가 멈추지 않게
            t_gate = BashOperator(
                task_id="gate",
                do_xcom_push=True,
                bash_command=f"""
                set +e
                cd /opt/airflow

                out=$(python /opt/airflow/tools/release_gate.py --model {m} 2>&1)
                rc=$?
                echo "$out"

                if echo "$out" | grep -q "PASS"; then
                  echo "PASS"
                else
                  # 짧은 사유: 마지막 줄만
                  reason=$(echo "$out" | tail -n 1)
                  echo "FAIL: $reason"
                fi

                exit 0
                """,
                env={"PYTHONUNBUFFERED": "1"},
            )

            # 4.5) gate 결과로 분기
            t_branch = BranchPythonOperator(
                task_id="branch_register",
                python_callable=branch_on_gate_pass,
                op_kwargs={"model": m},
            )

            t_skip = EmptyOperator(task_id="skip_register")

            # 5) register (PASS일 때만 실행되도록 branch로 제어)
            # register 자체 실패해도 전체 DAG는 계속 진행되게 exit 0 처리
            t_register = BashOperator(
                task_id="register",
                bash_command=f"""
                set +e
                cd /opt/airflow
                python /opt/airflow/cli/ct.py register --model {m}
                rc=$?
                if [ $rc -ne 0 ]; then
                  echo "[WARN] register failed ({m}) rc=$rc"
                fi
                exit 0
                """,
                env={
                    "PYTHONUNBUFFERED": "1",
                    "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
                },
            )

            t_join = EmptyOperator(
                task_id="join",
                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            )


            # 연결
            t_train >> t_eval >> t_gate >> t_branch
            t_branch >> t_register >> t_join
            t_branch >> t_skip >> t_join

        tg_end = t_join    # group의 종료점을 join으로

        # snapshot 이후 모델별 그룹 실행 (여기서는 리소스를 고려하여 순차 실행: baseline -> resnet34 -> efficientnet)
        # 병렬로 하고 싶으면 "tg"들을 snapshot에 각각 바로 붙이면 됨
        if prev_end is None:
            t_snapshot >> tg
        else:
            prev_end >> tg
        prev_end = tg_end
