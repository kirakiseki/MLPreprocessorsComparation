from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer
import time
import json
import os
import argparse

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", type=int, required=True, help="抽样比例，例如10表示10%")
parser.add_argument("--vocab_size", type=int, required=True, help="词汇表大小，例如100000")
args = parser.parse_args()
sample_rate = args.sample_rate
vocab_size = args.vocab_size

# 路径设置
file_path = os.path.dirname(__file__)
result_dir = os.path.join(
    file_path,
    "run",
    f"{sample_rate}pct_{vocab_size}vocab_{time.strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"result_{sample_rate}pct_{vocab_size}vocab.json")

# 初始化 Spark
spark = SparkSession.builder \
    .appName(f"exp1_minipile_countvectorizer_{sample_rate}pct_{vocab_size}vocab") \
    .config("spark.executor.memory", "31g") \
    .config("spark.executor.cores", "8") \
    .config("spark.driver.memory", "31g") \
    .getOrCreate()

# 获取 AppID
app_id = spark.sparkContext.applicationId

# =======================
# 数据加载
# =======================
data_path = f"s3a://bigdata-dataset/MiniPile/step_sampled/{sample_rate}"
df = spark.read.parquet(data_path).select("text")

# =======================
# 文本预处理（Spark MLlib）
# =======================

# 1. Tokenizer
tokenizer = Tokenizer(
    inputCol="text",
    outputCol="tokens"
)
df_tokens = tokenizer.transform(df).select("tokens")

# 预热缓存
df_tokens.cache().count()

# =======================
# CountVectorizer
# =======================
vectorizers = {
    "CountVectorizer": CountVectorizer(
        inputCol="tokens",
        outputCol="features",
        vocabSize=vocab_size,
        minDF=2              # 降低极低频词干扰
    )
}

results = {}

for name, vectorizer in vectorizers.items():
    times = []
    for run in range(3):
        start = time.time()

        model = vectorizer.fit(df_tokens)
        transformed = model.transform(df_tokens)

        # action 触发完整 DAG
        transformed.count()

        end = time.time()
        times.append(end - start)

    results[name] = {
        "avg_time": sum(times) / len(times),
        "raw_times": times,
        "vocab_size": model.vocabulary.__len__()
    }

# =======================
# 结果输出
# =======================
output = {
    "app_id": app_id,
    "dataset": "MiniPile",
    "operator": "CountVectorizer",
    "sample_rate": sample_rate,
    "results": results
}

with open(result_path, "w") as f:
    json.dump(output, f, indent=2)

spark.stop()

