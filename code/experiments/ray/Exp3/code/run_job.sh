#!/bin/bash
set -e

# -----------------------------
# 配置参数
# -----------------------------
CONTAINER_NAME="gdx_bigdata_ray_head"

EXPERIMENTS_DIR="/data/gdx/bigdata/experiments"
TASK_DIR="ray/exp3"
TASK_NAME="exp3"
LOCAL_WORKING_DIR="$EXPERIMENTS_DIR/$TASK_DIR"
CONTAINER_WORK_DIR="/home/ray/experiments/$TASK_DIR"

SCRIPT_NAME="main.py"
RAY_DASHBOARD="http://ray-head:8265"

RESET_SCRIPT="/data/gdx/bigdata/deploy/ray/reset_ray_cluster.sh"

# -----------------------------
# 循环提交不同抽样比例任务
# -----------------------------
for RATE in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;
do
    echo "重置Ray集群"
    bash $RESET_SCRIPT
    sleep 30

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULT_DIR="${CONTAINER_WORK_DIR}/run/${RATE}pct_${TIMESTAMP}"
    LOCAL_RESULT_DIR="${LOCAL_WORKING_DIR}/run/${RATE}pct_${TIMESTAMP}"

# -----------------------------
# Runtime Env
# -----------------------------
RUNTIME_ENV_JSON='{
    "pip": {
        "packages": [
            "pandas",
            "numpy",
            "s3fs",
            "fsspec",
            "boto3",
            "pyarrow",
            "psutil",
            "datasketches"
        ],
        "pip_install_options": ["--index-url=https://pypi.tuna.tsinghua.edu.cn/simple"]
    },
    "env_vars": {
        "AWS_ACCESS_KEY_ID": "dasebigdata",
        "AWS_SECRET_ACCESS_KEY": "dasebigdata",
        "AWS_ENDPOINT": "minio:9000",
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        "S3_USE_HTTPS": "false",
        "RAY_DATA_STRICT_MODE": "0",
        "RAY_PROFILING": "1",
        "RAY_task_events_report_interval_ms": "0",
        "WORKING_DIR": "'${CONTAINER_WORK_DIR}'",
        "RESULT_DIR": "'${RESULT_DIR}'"
    },
    "working_dir": "'${CONTAINER_WORK_DIR}'",
    "excludes": ["/run*"]
}'

    LOG_FILE="${LOCAL_RESULT_DIR}/submit_ray_job.txt"
    mkdir -p "${LOCAL_RESULT_DIR}"

    echo "=========================================="
    echo "提交 Ray 作业: 抽样比例 ${RATE}%"
    echo "结果目录: ${LOCAL_RESULT_DIR}"
    echo "=========================================="

{
    docker exec -i ${CONTAINER_NAME} /bin/bash << EOF
cd ${CONTAINER_WORK_DIR}

echo "提交 Ray 作业 -- sample_rate ${RATE} ..."

RAY_PROFILING=1 \
RAY_task_events_report_interval_ms=0 \
ray job submit \
    --address "${RAY_DASHBOARD}" \
    --working-dir "${CONTAINER_WORK_DIR}" \
    --runtime-env-json '${RUNTIME_ENV_JSON}' \
    -- python "${SCRIPT_NAME}" --sample_rate ${RATE} --task_name "${TASK_NAME}"
EOF

} 2>&1 | tee "${LOG_FILE}"

    echo "Ray 作业提交完成: 抽样比例 ${RATE}%"
    echo "日志已保存到: ${LOG_FILE}"

    echo "重启Ray集群并备份"
    bash $RESET_SCRIPT -s "exp3_${RATE}pct"


done

echo "=========================================="
echo "所有作业提交完成"
echo "=========================================="

