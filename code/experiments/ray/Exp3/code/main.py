import ray
import ray.data as rd
from ray.data.preprocessors import CustomKBinsDiscretizer
from ray.data.aggregate import ApproximateQuantile

import pyarrow.fs as fs
import argparse
import time
import json
import os
import sys
from datetime import datetime
import numpy as np
import psutil
import logging
from typing import Dict, Any, List, Tuple

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
parser.add_argument("--task_name", type=str, default="exp3")
parser.add_argument("--relative_error", type=float, default=0.1)
parser.add_argument("--num_buckets", type=int, default=10)
args = parser.parse_args()

sample_rate = args.sample_rate
task_name = args.task_name
relative_error = args.relative_error
num_buckets = args.num_buckets
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

data_path = f"s3://bigdata-dataset/Airline/step_sampled_finegrained/{sample_rate}"
output_base = f"bigdata-ray-logs/run_{task_name}_{sample_rate}pct_err{relative_error}_buckets{num_buckets}_{date_str}"
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
target_col = "ArrDelay"  # heavy-tail，最具代表性
ds = ds.select_columns([target_col])
ds = ds.materialize()
profiler.record_memory_usage("after_data_load")
timeline.record_cluster_state("after_data_load")
profiler.end_phase("data_load")

# ---------------------------
# 分位数计算与精度评估
# ---------------------------
profiler.start_phase("quantile_calculation")
timeline.record_cluster_state("before_quantile_calculation")

# 计算量化点 (1/num_buckets, 2/num_buckets, ..., (num_buckets-1)/num_buckets)
quantile_probs = [i / num_buckets for i in range(1, num_buckets)]
logger.info(f"计算分位数: {quantile_probs} (num_buckets={num_buckets})")

# 设置量化精度 (quantile_precision ≈ 1 / relative_error)
quantile_precision = max(10, min(10000, int(1 / relative_error)))
logger.info(f"使用 quantile_precision={quantile_precision} (relative_error={relative_error})")

# 获取基准分位数 (高精度)
base_quantile_precision = 10000  # 作为基准的高精度
gt_quantile_agg = ApproximateQuantile(
    on=target_col, 
    quantiles=quantile_probs, 
    quantile_precision=base_quantile_precision
)
gt_quantiles = ds.aggregate(gt_quantile_agg)[f"approx_quantile({target_col})"]

# 获取测试分位数 (指定精度)
approx_quantile_agg = ApproximateQuantile(
    on=target_col, 
    quantiles=quantile_probs, 
    quantile_precision=quantile_precision
)
approx_quantiles = ds.aggregate(approx_quantile_agg)[f"approx_quantile({target_col})"]

# 计算最小值和最大值 - 修正：min()和max()直接返回标量
min_val = ds.min(target_col)
max_val = ds.max(target_col)

# 构造分箱边界: [min, q1, q2, ..., max]
gt_bins = [min_val] + list(gt_quantiles) + [max_val]
approx_bins = [min_val] + list(approx_quantiles) + [max_val]

logger.info(f"基准分位数边界: {gt_bins}")
logger.info(f"测试分位数边界: {approx_bins}")

# 计算绝对误差
quantile_errors = [
    abs(float(a) - float(g)) for a, g in zip(approx_quantiles, gt_quantiles)
]

# 保存分位数统计
quantile_stats = {
    "quantile_probs": quantile_probs,
    "ground_truth_quantiles": [float(q) for q in gt_quantiles],
    "approx_quantiles": [float(q) for q in approx_quantiles],
    "abs_errors": quantile_errors,
    "min_value": float(min_val),
    "max_value": float(max_val),
    "gt_bins": [float(b) for b in gt_bins],
    "approx_bins": [float(b) for b in approx_bins],
    "base_quantile_precision": base_quantile_precision,
    "test_quantile_precision": quantile_precision
}

profiler.record_operator_stat("ApproximateQuantile", quantile_stats)
timeline.record_cluster_state("after_quantile_calculation")
profiler.end_phase("quantile_calculation")

# ---------------------------
# 等频分箱 (Quantile Discretization)
# ---------------------------
profiler.start_phase("quantile_discretization")
timeline.record_cluster_state("before_discretization")

# 创建分箱器 (使用测试分位数边界)
discretizer = CustomKBinsDiscretizer(
    columns=[target_col],
    bins={target_col: approx_bins},
    right=True,  # (a, b] 区间
    include_lowest=True,  # 包含最小值
    duplicates="drop",  # 处理重复边界
    output_columns=[f"{target_col}_bucket"]
)

# 应用分箱
start_time = time.time()
ds_binned = discretizer.transform(ds)
ds_binned.materialize()  # 触发计算
elapsed_time = time.time() - start_time

# 获取实际分箱数 (可能因重复边界而减少) - 修正：unique()返回列表，用len()获取长度
unique_buckets = ds_binned.unique(f"{target_col}_bucket")
actual_num_buckets = len(unique_buckets)
actual_bins = len(approx_bins) - 1  # 理论分箱数
if actual_num_buckets < actual_bins:
    logger.warning(f"实际分箱数 ({actual_num_buckets}) 小于理论值 ({actual_bins})，可能有重复边界")
    logger.warning(f"唯一桶值: {unique_buckets}")

# 保存分箱统计
discretizer_stats = {
    "num_buckets_requested": num_buckets,
    "num_buckets_actual": int(actual_num_buckets),
    "discretization_time_sec": elapsed_time,
    "bin_edges_used": [float(b) for b in approx_bins],
    "unique_bucket_values": [str(b) for b in unique_buckets],  # 转为字符串以便JSON序列化
    "right": True,
    "include_lowest": True,
    "duplicates": "drop"
}

profiler.record_operator_stat("CustomKBinsDiscretizer", discretizer_stats)
profiler.record_memory_usage("after_discretization")
timeline.record_cluster_state("after_discretization")
profiler.end_phase("quantile_discretization")

# ---------------------------
# 保存主要实验结果
# ---------------------------
output = {
    "experiment": "exp3_quantile_discretizer",
    "ray_version": ray.__version__,
    "sample_rate": sample_rate,
    "operator": "CustomKBinsDiscretizer",
    "params": {
        "num_buckets_requested": num_buckets,
        "relative_error": relative_error,
        "quantile_precision": quantile_precision,
        "target_column": target_col
    },
    "quantile_accuracy": {
        "quantile_probs": quantile_probs,
        "ground_truth_quantiles": gt_quantiles,
        "approx_quantiles": approx_quantiles,
        "abs_errors": quantile_errors,
        "min_value": min_val,
        "max_value": max_val,
        "gt_bins": gt_bins,
        "approx_bins": approx_bins
    },
    "performance": {
        "discretization_time_sec": elapsed_time,
        "actual_num_buckets": actual_num_buckets,
        "base_quantile_precision": base_quantile_precision,
        "test_quantile_precision": quantile_precision
    },
    "cluster_resources": ray.cluster_resources()
}

# ---------------------------
# 保存 Ray Timeline
# ---------------------------
timeline.record_driver_event("pipeline_end")

# 保存标准Ray Timeline
ray.timeline(timeline_path)
with s3_fs.open_output_stream(s3_timeline_path) as f_out, open(timeline_path, "rb") as f_in:
    f_out.write(f_in.read())

# 保存增强分布式 Timeline
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
with s3_fs.open_output_stream(result_path) as f:
    f.write(json.dumps(output, indent=2).encode("utf-8"))

logger.info(f"Ray 任务完成! 结果保存至: {result_path}")
logger.info(f"Timeline 保存至: {s3_timeline_path}")
logger.info(f"性能指标保存至: {profiler_s3_path}")

# ---------------------------
# 验证分箱结果 (可选)
# ---------------------------
if actual_num_buckets > 0:
    bucket_counts = ds_binned.groupby(f"{target_col}_bucket").count()
    logger.info(f"分箱分布:\n{bucket_counts.to_pandas()}")
else:
    logger.warning("分箱失败，无有效桶生成")

ray.shutdown()
