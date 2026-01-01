from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler
import time
import json
import os
import argparse

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", type=int, required=True, help="抽样比例，例如10表示10%")
args = parser.parse_args()
sample_rate = args.sample_rate

# 路径设置
file_path = os.path.dirname(__file__)
result_dir = os.path.join(file_path, "run", f"{sample_rate}pct_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"result_{sample_rate}pct.json")

# 初始化Spark
spark = SparkSession.builder \
    .appName(f"exp1_sample_{sample_rate}pct") \
    .config("spark.executor.memory", "31g") \
    .config("spark.executor.cores", "8") \
    .config("spark.driver.memory", "31g") \
    .getOrCreate()

# 获取AppID
app_id = spark.sparkContext.applicationId

# 数据加载
data_path = f"s3a://bigdata-dataset/Airline/step_sampled/{sample_rate}"
df = spark.read.parquet(data_path)

# 特征列
feature_cols = [
    "ArrDelay",
    "Distance",
    "AirTime",
    "TaxiOut",
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df).select("features")

# 预热缓存
df_vector.cache().count()

# 算子
scalers = {
    "StandardScaler": StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
}

results = {}

for name, scaler in scalers.items():
    times = []
    for run in range(3):
        start = time.time()
        model = scaler.fit(df_vector)
        transformed = model.transform(df_vector)
        transformed.count()
        end = time.time()
        times.append(end - start)
    results[name] = {
        "avg_time": sum(times)/len(times),
        "raw_times": times
    }

# 添加AppID和sample_rate
output = {
    "app_id": app_id,
    "sample_rate": sample_rate,
    "results": results
}

# 保存JSON
with open(result_path, "w") as f:
    json.dump(output, f, indent=2)

spark.stop()

