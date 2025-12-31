from pyspark.sql import SparkSession
from pyspark.ml.feature import QuantileDiscretizer
import time
import json
import os
import argparse

# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", type=int, required=True, help="抽样比例，例如10表示10%")
parser.add_argument("--relative_error", type=float, default=0.1, help="Spark approximateQuantile 的 relativeError")
parser.add_argument("--num_buckets", type=int, default=10, help="分箱数量")
args = parser.parse_args()

sample_rate = args.sample_rate
relative_error = args.relative_error
num_buckets = args.num_buckets

# =========================
# 路径设置
# =========================
file_path = os.path.dirname(__file__)
result_dir = os.path.join(
    file_path,
    "run",
    f"exp3_quantile_{sample_rate}pct_err{relative_error}_{time.strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, "result_exp3.json")

# =========================
# Spark 初始化
# =========================
spark = SparkSession.builder \
    .appName(f"exp3_quantile_{sample_rate}pct") \
    .config("spark.executor.memory", "31g") \
    .config("spark.executor.cores", "8") \
    .config("spark.driver.memory", "31g") \
    .getOrCreate()

app_id = spark.sparkContext.applicationId

# =========================
# 数据加载
# =========================
data_path = f"s3a://bigdata-dataset/Airline/step_sampled_finegrained/{sample_rate}"
df = spark.read.parquet(data_path)

target_col = "ArrDelay"  # heavy-tail，最具代表性
df_col = df.select(target_col).cache()
df_col.count()

# =========================
# 分位数精度评估
# =========================
quantile_probs = [0.9, 0.95, 0.99]

gt_quantiles = df_col.approxQuantile(target_col, quantile_probs, 0.0)
approx_quantiles = df_col.approxQuantile(target_col, quantile_probs, relative_error)

quantile_errors = {
    "probs": quantile_probs,
    "ground_truth": gt_quantiles,
    "approx": approx_quantiles,
    "abs_error": [
        abs(a - g) for a, g in zip(approx_quantiles, gt_quantiles)
    ]
}

# =========================
# QuantileDiscretizer 性能
# =========================
times = []
for run in range(3):
    discretizer = QuantileDiscretizer(
        inputCol=target_col,
        outputCol=f"{target_col}_bucket",
        numBuckets=num_buckets,
        relativeError=relative_error
    )

    start = time.time()
    model = discretizer.fit(df_col)
    out = model.transform(df_col)
    out.count()
    elapsed_time = time.time() - start
    times.append(elapsed_time)
    

avg_time = sum(times) / len(times)

# =========================
# 结果输出
# =========================
output = {
    "experiment": "exp3_quantile_discretizer",
    "app_id": app_id,
    "spark_version": spark.version,
    "sample_rate": sample_rate,
    "operator": "QuantileDiscretizer",
    "params": {
        "num_buckets": num_buckets,
        "relative_error": relative_error
    },
    "quantile_definition": {
        "method": "Spark approxQuantile",
        "exact_baseline": "relativeError = 0"
    },
    "quantile_accuracy": quantile_errors,
    "performance": {
        "fit_transform_time_sec": avg_time
    }
}

with open(result_path, "w") as f:
    json.dump(output, f, indent=2)

spark.stop()

