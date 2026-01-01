#!/bin/bash

set -e

# 配置参数
CONTAINER_NAME="gdx_bigdata_spark_master"
SPARK_HOME="/opt/spark"
MASTER="spark://spark-master:7077"
PYSPARK_SCRIPT="main.py"
EXPERIMENTS_DIR="/data/gdx/bigdata/experiments"
TASK_DIR="spark/exp3"
LOCAL_WORK_DIR="$EXPERIMENTS_DIR/$TASK_DIR"
CONTAINER_WORK_DIR="/opt/spark/work-dir/$TASK_DIR"

RESET_SCRIPT="/data/gdx/bigdata/deploy/spark/reset_spark_cluster.sh"

for RATE in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;
do

echo "=========================================="
echo "提交Spark作业到集群"
echo "=========================================="
echo "容器名称: $CONTAINER_NAME"
echo "Spark Master: $MASTER"
echo "脚本: $PYSPARK_SCRIPT"
echo "本地工作目录: $LOCAL_WORK_DIR"
echo "容器内工作目录: $CONTAINER_WORK_DIR"
echo "=========================================="

echo "重置Spark集群"
bash $RESET_SCRIPT
sleep 30

LOG_FILE="$LOCAL_WORK_DIR/run/logs/exp3_run_${RATE}pct_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

{
# 提交Spark作业
docker exec -i $CONTAINER_NAME /bin/bash << EOF
    cd $CONTAINER_WORK_DIR
    
    echo "当前工作目录: \$(pwd)"
    echo "文件列表:"
    ls -la
    
    echo "提交Spark作业..."
    $SPARK_HOME/bin/spark-submit \
        --master $MASTER \
        --deploy-mode client \
        $PYSPARK_SCRIPT --sample_rate $RATE
    
    echo "Spark作业 AppID: \$APP_ID"
    
    EXIT_CODE=\$?
    echo "Spark作业执行完成，退出代码: \$EXIT_CODE"
EOF
} 2>&1 | tee "${LOG_FILE}"

echo "重启Spark集群并备份"
bash $RESET_SCRIPT -s "exp3_${RATE}pct"

done

echo "=========================================="
echo "作业提交完成"
echo "=========================================="
