#!/bin/bash

set -e

# 配置参数
CONTAINER_NAME="bigdata_spark_master"
SPARK_HOME="/opt/spark"
MASTER="spark://spark-master:7077"
PYSPARK_SCRIPT="count_merge.py"
EXPERIMENTS_DIR="/root/BigData/experiments"
TASK_DIR="spark/merge"
LOCAL_WORK_DIR="$EXPERIMENTS_DIR/$TASK_DIR"
CONTAINER_WORK_DIR="/opt/spark/work-dir/$TASK_DIR"

# 检查容器是否运行
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "错误: 容器 '$CONTAINER_NAME' 未运行"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$LOCAL_WORK_DIR/$PYSPARK_SCRIPT" ]; then
    echo "错误: Python脚本 '$PYSPARK_SCRIPT' 不存在于 $LOCAL_WORK_DIR/"
    exit 1
fi

echo "=========================================="
echo "提交Spark作业到集群"
echo "=========================================="
echo "容器名称: $CONTAINER_NAME"
echo "Spark Master: $MASTER"
echo "脚本: $PYSPARK_SCRIPT"
echo "本地工作目录: $LOCAL_WORK_DIR"
echo "容器内工作目录: $CONTAINER_WORK_DIR"
echo "=========================================="

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
        $PYSPARK_SCRIPT
    
    EXIT_CODE=\$?
    echo "Spark作业执行完成，退出代码: \$EXIT_CODE"
    
    # 显示结果文件
    if [ -f "result.txt" ]; then
        echo "=========================================="
        echo "结果文件内容:"
        echo "=========================================="
        cat result.txt
    fi
EOF

echo "=========================================="
echo "作业提交完成"
echo "=========================================="