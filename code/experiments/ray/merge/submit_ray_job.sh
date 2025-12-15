#!/bin/bash

# submit_ray_job.sh
# 在bigdata_ray_head容器内提交Ray作业

set -e  # 遇到错误立即退出

# 容器名称
CONTAINER_NAME="bigdata_ray_head"

# 检查容器是否在运行
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "错误: 容器 ${CONTAINER_NAME} 未运行"
    exit 1
fi

# 进入容器并执行命令
echo "进入容器: ${CONTAINER_NAME}"
echo "激活base环境并提交Ray作业..."

# 定义工作目录和脚本路径
WORKING_DIR="/home/ray/experiments/ray/merge"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${WORKING_DIR}/run/${TIMESTAMP}"
LOCAL_RESULT_DIR="./run/${TIMESTAMP}"

mkdir -p "${LOCAL_RESULT_DIR}"

SCRIPT_NAME="merge_csv_ray.py"
SCRIPT_PATH="${WORKING_DIR}/${SCRIPT_NAME}"

# 检查脚本是否存在
echo "检查脚本是否存在: ${SCRIPT_PATH}"
docker exec ${CONTAINER_NAME} /bin/bash -c "
    if [ ! -f \"${SCRIPT_PATH}\" ]; then
        echo \"错误: 脚本 ${SCRIPT_PATH} 不存在\"
        exit 1
    fi
    echo \"脚本找到: ${SCRIPT_PATH}\"
"

# Ray Dashboard地址
RAY_DASHBOARD_ADDRESS="http://ray-head:8265"

# 创建运行时环境JSON
RUNTIME_ENV_JSON='{
    "pip": {
        "packages": [
            "pandas==2.1.4",
            "numpy==1.24.3",
            "s3fs==2023.12.2",
            "fsspec==2023.12.2",
            "boto3==1.34.17",
            "pyarrow==14.0.2",
            "psutil==5.9.7"
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
        "WORKING_DIR": "'${WORKING_DIR}'",
        "RESULT_DIR": "'${RESULT_DIR}'"
    }
}'

LOG_FILE="${LOCAL_RESULT_DIR}/submit_ray_job.txt"


echo "=========================================="
echo "准备提交Ray作业..."
echo "工作目录: ${WORKING_DIR}"
echo "结果目录: ${RESULT_DIR}"
echo "本地结果目录: ${LOCAL_RESULT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "脚本名称: ${SCRIPT_NAME}"
echo "Ray Dashboard: ${RAY_DASHBOARD_ADDRESS}"
echo "=========================================="


# 进入容器并提交作业
docker exec -it ${CONTAINER_NAME} /bin/bash -c "
    echo \"当前Python路径: \$(which python)\"
    echo \"Python版本: \$(python --version)\"
    echo \"Ray版本: \$(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo '未安装')\"
    
    # 检查Ray集群状态
    echo \"检查Ray集群状态...\"
    ray status --address=ray-head:6379 || echo \"警告: 无法连接Ray集群\"
    
    # 提交作业
    echo \"正在提交Ray作业...\"
    
    RAY_PROFILING=1 \
    RAY_task_events_report_interval_ms=0 \
    ray job submit \
        --address=\"${RAY_DASHBOARD_ADDRESS}\" \
        --working-dir=\"${WORKING_DIR}\" \
        --runtime-env-json='${RUNTIME_ENV_JSON}' \
        -- \
        python \"${SCRIPT_NAME}\"
    
    echo \"作业提交完成!\"
    echo \"可以在Ray Dashboard查看作业状态: ${RAY_DASHBOARD_ADDRESS}\"
" 2>&1 | tee "${LOG_FILE}"

# 检查作业状态
echo ""
echo "=========================================="
echo "作业提交完成!"
echo ""
echo "查看作业状态的方法:"
echo "1. 访问Ray Dashboard: ${RAY_DASHBOARD_ADDRESS}"
echo "2. 查看日志文件: ${WORKING_DIR}/run/YYYYMMDD_HHMMSS/"
echo ""
echo "要实时查看作业日志，可以运行:"
echo "docker exec ${CONTAINER_NAME} tail -f /home/ray/experiments/ray/merge/run/*/result.txt"
echo "=========================================="