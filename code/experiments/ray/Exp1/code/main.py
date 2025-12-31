import ray
import ray.data as rd
from ray.data.preprocessors import StandardScaler

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
                'duration_seconds': duration,
                'start_time': self.start_times[phase_name],
                'end_time': time.time()
            }
            logger.info(f"阶段 {phase_name} 完成, 耗时: {duration:.2f}秒")
            return duration
        return 0.0

    def record_memory_usage(self, label: str):
        process = psutil.Process()
        mem = process.memory_info()
        self.memory_usage.append({
            'timestamp': time.time(),
            'label': label,
            'rss_mb': mem.rss / 1024 / 1024,
            'vms_mb': mem.vms / 1024 / 1024
        })

    def record_operator_stat(self, operator_name: str, stats: Dict[str, Any]):
        self.operator_stats[operator_name] = {
            'recorded_time': time.time(),
            **stats
        }

    def save_results(self, result_dir: str) -> str:
        results = {
            'overall_metrics': self.metrics,
            'operator_stats': self.operator_stats,
            'memory_usage': self.memory_usage,
            'system_info': {
                'ray_version': ray.__version__,
                'python_version': sys.version,
                'num_cpus': os.cpu_count(),
                'ray_cluster_resources': ray.cluster_resources() if ray.is_initialized() else {}
            }
        }
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, 'performance_metrics.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"性能指标已保存到: {result_file}")
        return result_file

# ---------------------------
# DistributedRayTimeline
# ---------------------------
class DistributedRayTimeline:
    def __init__(self):
        self.events: List[Dict] = []

    def record_driver_event(self, name: str, phase: str = "X"):
        event = {
            "cat": "driver",
            "name": name,
            "pid": "ray-head",
            "tid": "driver",
            "ts": time.time() * 1e6,
            "ph": phase,
            "args": {}
        }
        self.events.append(event)

    def record_worker_task(self, task_name: str, node_id: str, start_ts: float, end_ts: float, extra: Dict = None):
        event = {
            "cat": "worker_task",
            "name": task_name,
            "pid": node_id,
            "tid": f"task-{task_name}",
            "ts": start_ts * 1e6,
            "dur": (end_ts - start_ts) * 1e6,
            "ph": "X",
            "args": extra or {}
        }
        self.events.append(event)

    def record_cluster_state(self, label: str):
        nodes_info = []
        for n in list_nodes():
            nodes_info.append({
                "node_id": n["node_id"],
                "node_ip": n["node_ip"],
                "is_head_node": n["is_head_node"],
                "state": n["state"],
                "state_message": n["state_message"],
                "node_name": n["node_name"],
                "resources_total": n["resources_total"],
                "labels": n["labels"],
                "start_time_ms": n["start_time_ms"]
            })
        event = {
            "cat": "cluster_state",
            "name": label,
            "pid": "cluster",
            "tid": label,
            "ts": time.time() * 1e6,
            "ph": "i",
            "args": {
                "nodes": nodes_info,
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources()
            }
        }
        self.events.append(event)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.events, f, indent=2)
        logger.info(f"[DistributedRayTimeline] 保存到 {file_path}")

# ---------------------------
# 参数解析
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", type=int, required=True)
parser.add_argument("--task_name", type=str, default="exp1")
args = parser.parse_args()

sample_rate = args.sample_rate
task_name = args.task_name
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# S3 / MinIO 配置
# ---------------------------
s3_fs = fs.S3FileSystem(
    endpoint_override="minio:9000",
    access_key="dasebigdata",
    secret_key="dasebigdata",
    scheme="http"
)

data_path = f"s3://bigdata-dataset/Airline/step_sampled/{sample_rate}"
output_base = f"bigdata-ray-logs/run_{task_name}_{sample_rate}pct_{date_str}"
result_path = f"{output_base}/result_{sample_rate}pct.json"
timeline_path = f"/tmp/ray_timeline.json"
s3_timeline_path = f"{output_base}/ray_timeline.json"
profiler_local_dir = "/tmp/ray_profiler"
profiler_s3_path = f"{output_base}/performance_metrics.json"
distributed_timeline_path = f"/tmp/distributed_ray_timeline.json"
distributed_timeline_s3_path = f"{output_base}/distributed_ray_timeline.json"

# ---------------------------
# 初始化 Ray
# ---------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------
# 初始化 Profiler 和 Timeline
# ---------------------------
profiler = RayProfiler()
timeline = DistributedRayTimeline()

timeline.record_driver_event("pipeline_start")
timeline.record_cluster_state("before_data_load")

# ---------------------------
# 数据加载阶段
# ---------------------------
profiler.start_phase("data_load")
ds = rd.read_parquet(data_path, filesystem=s3_fs)
feature_cols = ["ArrDelay", "Distance", "AirTime", "TaxiOut"]
output_cols = [f"scaled_{c}" for c in feature_cols]

ds = ds.select_columns(feature_cols)
ds = ds.materialize()
profiler.record_memory_usage("after_data_load")
timeline.record_cluster_state("after_data_load")
profiler.end_phase("data_load")

# ---------------------------
# 标准化阶段
# ---------------------------
scaler = StandardScaler(columns=feature_cols, output_columns=output_cols)
times = []

profiler.start_phase("standard_scaler_fit_transform")
timeline.record_cluster_state("before_scaling")
for _ in range(3):
    start = time.time()
    ds_scaled = scaler.fit_transform(ds)
    ds_scaled.materialize()
    end = time.time()
    times.append(end - start)

profiler.record_memory_usage("after_scaling")

# 保存算子 DAG stats
dag_stats = ds_scaled.stats()
profiler.record_operator_stat("StandardScaler", {
    "fit_transform_times": times,
    "dag_stats": dag_stats
})

# ---------------------------
# 利用 State API 增强调度信息
# ---------------------------
running_tasks = list_tasks(filters=[("state", "=", "RUNNING")], detail=True)
for t in running_tasks:
    timeline.record_worker_task(
        task_name=t.get("name", "unknown"),
        node_id=t.get("node_id", "unknown"),
        start_ts=t.get("start_time", time.time()),
        end_ts=t.get("end_time", time.time()),
        extra={
            "task_id": t.get("task_id"),
            "scheduling_state": t.get("scheduling_state"),
            "runtime_env": t.get("runtime_env"),
        }
    )

timeline.record_cluster_state("after_scaling")
profiler.end_phase("standard_scaler_fit_transform")

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

