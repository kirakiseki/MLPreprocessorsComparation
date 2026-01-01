import ray
import ray.data as rd
from ray.data.preprocessors import CountVectorizer

import pyarrow.fs as fs
import argparse
import time
import json
import os
import sys
from datetime import datetime
import psutil
import logging
from typing import Dict, Any, List

# State API
from ray.util.state import list_tasks, list_nodes

# ---------------------------
# 日志配置
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# RayProfiler
# ---------------------------
class RayProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.operator_stats = {}
        self.memory_usage = []

    def start_phase(self, phase_name: str):
        self.start_times[phase_name] = time.time()
        logger.info(f"开始阶段: {phase_name}")

    def end_phase(self, phase_name: str) -> float:
        if phase_name in self.start_times:
            duration = time.time() - self.start_times[phase_name]
            self.metrics[phase_name] = {
                "duration_seconds": duration,
                "start_time": self.start_times[phase_name],
                "end_time": time.time()
            }
            logger.info(f"阶段 {phase_name} 完成, 耗时: {duration:.2f}秒")
            return duration
        return 0.0

    def record_memory_usage(self, label: str):
        process = psutil.Process()
        mem = process.memory_info()
        self.memory_usage.append({
            "timestamp": time.time(),
            "label": label,
            "rss_mb": mem.rss / 1024 / 1024,
            "vms_mb": mem.vms / 1024 / 1024
        })

    def record_operator_stat(self, operator_name: str, stats: Dict[str, Any]):
        self.operator_stats[operator_name] = {
            "recorded_time": time.time(),
            **stats
        }

    def save_results(self, result_dir: str) -> str:
        results = {
            "overall_metrics": self.metrics,
            "operator_stats": self.operator_stats,
            "memory_usage": self.memory_usage,
            "system_info": {
                "ray_version": ray.__version__,
                "python_version": sys.version,
                "num_cpus": os.cpu_count(),
                "ray_cluster_resources": ray.cluster_resources()
            }
        }
        os.makedirs(result_dir, exist_ok=True)
        path = os.path.join(result_dir, "performance_metrics.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path


# ---------------------------
# DistributedRayTimeline
# ---------------------------
class DistributedRayTimeline:
    def __init__(self):
        self.events: List[Dict] = []

    def record_driver_event(self, name: str, phase: str = "X"):
        self.events.append({
            "cat": "driver",
            "name": name,
            "pid": "ray-head",
            "tid": "driver",
            "ts": time.time() * 1e6,
            "ph": phase,
            "args": {}
        })

    def record_worker_task(self, task_name: str, node_id: str, start_ts: float, end_ts: float, extra: Dict = None):
        self.events.append({
            "cat": "worker_task",
            "name": task_name,
            "pid": node_id,
            "tid": f"task-{task_name}",
            "ts": start_ts * 1e6,
            "dur": (end_ts - start_ts) * 1e6,
            "ph": "X",
            "args": extra or {}
        })

    def record_cluster_state(self, label: str):
        nodes_info = []
        for n in list_nodes():
            nodes_info.append({
                "node_id": n["node_id"],
                "node_ip": n["node_ip"],
                "is_head_node": n["is_head_node"],
                "state": n["state"],
                "resources_total": n["resources_total"]
            })
        self.events.append({
            "cat": "cluster_state",
            "name": label,
            "pid": "cluster",
            "tid": label,
            "ts": time.time() * 1e6,
            "ph": "i",
            "args": {
                "nodes": nodes_info,
                "cluster_resources": ray.cluster_resources()
            }
        })

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.events, f, indent=2)


# ---------------------------
# 参数解析
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", type=int, required=True)
parser.add_argument("--vocab_size", type=int, required=True)
parser.add_argument("--task_name", type=str, default="exp2")
args = parser.parse_args()

sample_rate = args.sample_rate
vocab_size = args.vocab_size
task_name = args.task_name
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# S3 配置
# ---------------------------
s3_fs = fs.S3FileSystem(
    endpoint_override="minio:9000",
    access_key="dasebigdata",
    secret_key="dasebigdata",
    scheme="http"
)

data_path = f"s3://bigdata-dataset/MiniPile/step_sampled/{sample_rate}"
output_base = f"bigdata-ray-logs/run_{task_name}_{sample_rate}pct_{vocab_size}vocab_{date_str}"
result_path = f"{output_base}/result_{sample_rate}pct_{vocab_size}vocab.json"
timeline_path = "/tmp/ray_timeline.json"
s3_timeline_path = f"{output_base}/ray_timeline.json"
profiler_local_dir = "/tmp/ray_profiler"
profiler_s3_path = f"{output_base}/performance_metrics.json"
distributed_timeline_path = f"/tmp/distributed_ray_timeline.json"
distributed_timeline_s3_path = f"{output_base}/distributed_ray_timeline.json"

# ---------------------------
# 初始化 Ray
# ---------------------------
ray.init(ignore_reinit_error=True)

profiler = RayProfiler()
timeline = DistributedRayTimeline()

timeline.record_driver_event("pipeline_start")
timeline.record_cluster_state("before_data_load")

# =======================
# 数据加载
# =======================
profiler.start_phase("data_load")
ds = rd.read_parquet(data_path, filesystem=s3_fs).select_columns(["text"])
ds = ds.materialize()
profiler.record_memory_usage("after_data_load")
timeline.record_cluster_state("after_data_load")
profiler.end_phase("data_load")

# =======================
# CountVectorizer
# =======================
vectorizer = CountVectorizer(
    columns=["text"],
    output_columns=["features"],
    max_features=vocab_size
)


times = []

profiler.start_phase("countvectorizer_fit_transform")
timeline.record_cluster_state("before_vectorize")

for _ in range(3):
    start = time.time()
    ds_vec = vectorizer.fit_transform(ds)
    ds_vec.materialize()  
    end = time.time()
    times.append(end - start)

profiler.record_memory_usage("after_vectorize")

token_stat_key = "token_counts(text)"
if token_stat_key in vectorizer.stats_:
    vocab_size_actual = len(vectorizer.stats_[token_stat_key])
else:
    vocab_size_actual = None

profiler.record_operator_stat(
    "CountVectorizer",
    {
        "fit_transform_times": times,
        "vocab_size_actual": vocab_size_actual,
        "vocab_size_configured": vocab_size,
    }
)


timeline.record_cluster_state("after_vectorize")
profiler.end_phase("countvectorizer_fit_transform")


# ---------------------------
# 保存 Ray Timeline
# ---------------------------
ray.timeline(timeline_path)
with s3_fs.open_output_stream(s3_timeline_path) as f_out, open(timeline_path, "rb") as f_in:
    f_out.write(f_in.read())

# ---------------------------
# 保存增强分布式 Timeline
# ---------------------------
timeline.record_driver_event("pipeline_end")
timeline.save(distributed_timeline_path)
with s3_fs.open_output_stream(distributed_timeline_s3_path) as f_out, open(distributed_timeline_path, "rb") as f_in:
    f_out.write(f_in.read())

# ---------------------------
# 保存 Profiler 分析结果
# ---------------------------
os.makedirs(profiler_local_dir, exist_ok=True)
profiler_file = profiler.save_results(profiler_local_dir)
with s3_fs.open_output_stream(profiler_s3_path) as f_out, open(profiler_file, "rb") as f_in:
    f_out.write(f_in.read())

# ---------------------------
# 保存主要实验结果
# ---------------------------
output = {
    "sample_rate": sample_rate,
    "task_name": task_name,
    "results": {
        "StandardScaler": {
            "avg_time": sum(times) / len(times),
            "raw_times": times
        }
    }
}

with s3_fs.open_output_stream(result_path) as f:
    f.write(json.dumps(output, indent=2).encode("utf-8"))

logger.info("Ray 分布式任务完成")

