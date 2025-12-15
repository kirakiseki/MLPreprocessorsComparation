import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import ray
from ray.data import Dataset, read_csv
from ray.data.context import DatasetContext
import psutil

from pyarrow import fs

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RayProfiler:
    """自定义Ray性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.operator_stats = {}
        self.memory_usage = []
        
    def start_phase(self, phase_name: str):
        """开始一个执行阶段"""
        self.start_times[phase_name] = time.time()
        logger.info(f"开始阶段: {phase_name}")
        
    def end_phase(self, phase_name: str) -> float:
        """结束一个执行阶段并返回耗时"""
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
        """记录内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage.append({
            'timestamp': time.time(),
            'label': label,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
        })
    
    def record_operator_stat(self, operator_name: str, stats: Dict[str, Any]):
        """记录算子统计信息"""
        self.operator_stats[operator_name] = {
            'recorded_time': time.time(),
            **stats
        }
    
    def save_results(self, result_dir: str):
        """保存性能分析结果"""
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
        
        result_file = os.path.join(result_dir, 'performance_metrics.json')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"性能指标已保存到: {result_file}")
        return result_file

def load_dataset_with_ray_data(path: str, label: str, profiler: RayProfiler) -> Optional[Dataset]:
    """使用Ray Data加载数据集"""
    logger.info(f"使用Ray Data加载 {label}: {path}")
    
    s3_fs = fs.S3FileSystem(
        endpoint_override='minio:9000', #
        access_key='dasebigdata',       
        secret_key='dasebigdata',       
        scheme='http'                       
    )
    
    try:
        
        ds = read_csv(
            path,
            filesystem=s3_fs,
        )
        
        # 立即执行一个操作以触发加载
        count = ds.count()
        
        logger.info(f"{label} 加载完成: {count} 行")
        
        profiler.record_operator_stat(f"load_{label}", {
            'path': path,
            'rows': count,
            'schema': str(ds.schema()) if ds.schema() else 'None'
        })
        
        return ds
        
    except Exception as e:
        logger.error(f"Ray Data加载 {label} 失败: {str(e)}")
        # 创建空数据集作为后备
        logger.info(f"为 {label} 创建空数据集作为后备")
        return create_empty_dataset()

def create_empty_dataset() -> Dataset:
    """创建空数据集"""
    # 创建一个包含一个空行的数据集
    return ray.data.from_items([{"_empty": True}])

def analyze_dataset(ds: Dataset, label: str, profiler: RayProfiler) -> Dict[str, Any]:
    """分析数据集并收集统计信息"""
    start_time = time.time()
    
    # 收集基础统计信息
    stats = {
        'label': label,
    }
    
    try:
        # 获取schema
        schema = ds.schema()
        if schema:
            stats['schema'] = str(schema)
            stats['column_names'] = list(schema.names)
        
        # 统计行数
        count = ds.count()
        stats['row_count'] = count
        
        # 检查是否为空数据集
        if count == 1:
            # 检查是否是我们创建的空数据集
            sample = ds.take(1)
            if sample and "_empty" in sample[0]:
                stats['is_empty'] = True
                stats['row_count'] = 0
        
        # 获取内存使用估计
        try:
            if not stats.get('is_empty', False):
                size_bytes = ds.size_bytes()
                stats['size_bytes_estimate'] = size_bytes
                stats['size_mb_estimate'] = size_bytes / (1024 * 1024)
            else:
                stats['size_bytes_estimate'] = 0
                stats['size_mb_estimate'] = 0
        except Exception as e:
            stats['size_bytes_estimate'] = None
            stats['size_mb_estimate'] = None
            
        # 获取数据采样（如果不是空数据集）
        if not stats.get('is_empty', False) and stats.get('row_count', 0) > 0:
            sample = ds.take(1)
            if sample:
                stats['sample_row'] = str(sample[0])
            
    except Exception as e:
        logger.warning(f"分析数据集 {label} 时出错: {str(e)}")
        stats['error'] = str(e)
    
    duration = time.time() - start_time
    stats['analysis_time'] = duration
    
    profiler.record_operator_stat(f"analyze_{label}", {
        'duration': duration,
        'row_count': stats.get('row_count', 0),
        'size_mb': stats.get('size_mb_estimate', 0),
        'is_empty': stats.get('is_empty', False)
    })
    
    return stats

def main():
    """主函数"""
    
    # 创建结果目录
    working_dir = ray.get_runtime_context().runtime_env.env_vars().get("WORKING_DIR", ".")
    print(f"工作目录: {working_dir}")
    
    # 确保result_dir存在
    result_dir = ray.get_runtime_context().runtime_env.env_vars().get("RESULT_DIR", ".")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "result.txt")
    
    logger.info(f"结果将保存到: {result_path}")
    
    # 初始化性能分析器
    profiler = RayProfiler()
    
    # 记录初始内存使用
    profiler.record_memory_usage("initial")
    
    # 初始化Ray（如果尚未初始化）
    if not ray.is_initialized():
        profiler.start_phase("ray_init")
        ray.init(ignore_reinit_error=True)
        logger.info(f"Ray集群资源: {ray.cluster_resources()}")
        profiler.end_phase("ray_init")
    
    profiler.record_memory_usage("after_ray_init")
    
    # 开始时间线记录
    profiler.start_phase("timeline_recording")
    timeline_dir = result_dir
    timeline_file = os.path.join(timeline_dir, "ray_timeline.json")
    logger.info(f"时间线记录将保存到: {timeline_dir}")
    
    # 定义S3路径
    training_path = "s3://bigdata-dataset/GiveMeSomeCredit/cs-training.csv"
    test_path = "s3://bigdata-dataset/GiveMeSomeCredit/cs-test.csv"
    
    # 使用Ray Data加载数据
    profiler.start_phase("data_loading")
    
    # 加载训练数据集
    training_ds = load_dataset_with_ray_data(training_path, "training_data", profiler)
    
    # 加载测试数据集
    test_ds = load_dataset_with_ray_data(test_path, "test_data", profiler)
    
    profiler.end_phase("data_loading")
    profiler.record_memory_usage("after_data_loading")
    
    # 分析数据集
    profiler.start_phase("dataset_analysis")
    training_stats = analyze_dataset(training_ds, "training", profiler)
    test_stats = analyze_dataset(test_ds, "test", profiler)
    profiler.end_phase("dataset_analysis")
    
    # 合并数据集
    profiler.start_phase("dataset_union")
    logger.info("开始合并数据集...")
    
    try:
        # 如果数据集为空，不进行合并操作
        if training_stats.get('is_empty', False) and test_stats.get('is_empty', False):
            logger.info("两个数据集都为空，跳过合并")
            combined_ds = create_empty_dataset()
            combined_count = 0
            combined_stats = {'row_count': 0, 'is_empty': True}
        else:
            # 执行合并操作
            combined_ds = training_ds.union(test_ds)
            
            # 触发计算以获取准确统计
            profiler.start_phase("trigger_computation")
            
            combined_count = combined_ds.count()
            combined_stats = analyze_dataset(combined_ds, "combined", profiler)
            
            profiler.end_phase("trigger_computation")
        
        profiler.end_phase("dataset_union")
        
    except Exception as e:
        logger.error(f"数据集合并失败: {e}")
        # 如果合并失败，尝试手动计算
        combined_count = training_stats.get('row_count', 0) + test_stats.get('row_count', 0)
        combined_stats = {'row_count': combined_count, 'error': str(e)}
        combined_ds = create_empty_dataset()
        profiler.end_phase("dataset_union")
    
    profiler.record_memory_usage("after_dataset_union")
    
    # 执行一些数据转换操作以测试性能
    profiler.start_phase("data_transformations")
    try:
        # 如果不是空数据集，添加一个简单的转换操作
        if not combined_stats.get('is_empty', False) and combined_stats.get('row_count', 0) > 0:
            transformed_ds = combined_ds.map_batches(
                lambda df: df,  # 这里可以添加实际的数据转换逻辑
                batch_format="pandas"
            )
            
            # 触发转换计算
            transformed_count = transformed_ds.count()
            profiler.record_operator_stat("transformation", {
                'input_rows': combined_count,
                'output_rows': transformed_count,
                'transformation_applied': 'identity_map'
            })
            
            logger.info(f"数据转换完成，转换后行数: {transformed_count}")
        else:
            logger.info("数据集为空，跳过数据转换")
    except Exception as e:
        logger.warning(f"数据转换失败: {e}")
    profiler.end_phase("data_transformations")
    
    # 收集详细的算子信息
    profiler.start_phase("collect_operator_stats")
    
    try:
        # 尝试导入ray.util.state
        from ray.util.state import list_tasks
        
        tasks = list_tasks(detail=True)
        
        operator_details = []
        for task in tasks:
            operator_details.append({
                'task_id': task.task_id,
                'name': task.name,
                'state': task.state,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'duration_ms': (task.end_time - task.start_time) * 1000 if task.end_time and task.start_time else None,
                'node_id': task.node_id,
                'worker_id': task.worker_id,
            })
        
        profiler.record_operator_stat("task_execution_details", {
            'num_tasks': len(operator_details),
            'tasks': operator_details
        })
        
    except Exception as e:
        logger.warning(f"无法获取任务统计信息: {e}")
    
    profiler.end_phase("collect_operator_stats")
    
    # 停止时间线记录
    profiler.start_phase("save_timeline")
    try:
        ray.timeline(filename=timeline_file)
        logger.info(f"时间线已保存到: {timeline_file}")
    except Exception as e:
        logger.warning(f"保存时间线失败: {e}")
    profiler.end_phase("save_timeline")
    
    # 写入结果
    profiler.start_phase("write_results")
    try:
        with open(result_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CSV文件合并统计结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结果目录: {result_dir}\n\n")
            
            f.write("训练数据集统计:\n")
            f.write(f"  文件路径: {training_path}\n")
            f.write(f"  行数: {training_stats.get('row_count', 0)}\n")
            if training_stats.get('is_empty', False):
                f.write("  状态: 空数据集（加载失败或桶不存在）\n")
            if 'column_names' in training_stats:
                f.write(f"  列数: {len(training_stats['column_names'])}\n")
                if len(training_stats['column_names']) > 0 and not training_stats.get('is_empty', False):
                    f.write(f"  列名: {', '.join(training_stats['column_names'][:10])}")
                    if len(training_stats['column_names']) > 10:
                        f.write(f", ... ({len(training_stats['column_names'])-10} more)")
                    f.write("\n")
            if 'size_mb_estimate' in training_stats and training_stats['size_mb_estimate']:
                f.write(f"  估计大小: {training_stats['size_mb_estimate']:.2f} MB\n")
            f.write("\n")
            
            f.write("测试数据集统计:\n")
            f.write(f"  文件路径: {test_path}\n")
            f.write(f"  行数: {test_stats.get('row_count', 0)}\n")
            if test_stats.get('is_empty', False):
                f.write("  状态: 空数据集（加载失败或桶不存在）\n")
            if 'column_names' in test_stats:
                f.write(f"  列数: {len(test_stats['column_names'])}\n")
                if len(test_stats['column_names']) > 0 and not test_stats.get('is_empty', False):
                    f.write(f"  列名: {', '.join(test_stats['column_names'][:10])}")
                    if len(test_stats['column_names']) > 10:
                        f.write(f", ... ({len(test_stats['column_names'])-10} more)")
                    f.write("\n")
            if 'size_mb_estimate' in test_stats and test_stats['size_mb_estimate']:
                f.write(f"  估计大小: {test_stats['size_mb_estimate']:.2f} MB\n")
            f.write("\n")
            
            f.write("合并后数据集统计:\n")
            f.write(f"  总行数: {combined_stats.get('row_count', 0)}\n")
            if combined_stats.get('is_empty', False):
                f.write("  状态: 空数据集\n")
            if 'size_mb_estimate' in combined_stats and combined_stats['size_mb_estimate']:
                f.write(f"  估计总大小: {combined_stats['size_mb_estimate']:.2f} MB\n")
            if 'error' in combined_stats:
                f.write(f"  错误信息: {combined_stats['error']}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("性能指标摘要:\n")
            f.write("=" * 80 + "\n\n")
            
            for phase, metrics in profiler.metrics.items():
                f.write(f"{phase}:\n")
                f.write(f"  耗时: {metrics.get('duration_seconds', 0):.2f} 秒\n")
                start_time = metrics.get('start_time', 0)
                end_time = metrics.get('end_time', 0)
                if start_time:
                    f.write(f"  开始时间: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}\n")
                if end_time:
                    f.write(f"  结束时间: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}\n")
                f.write("\n")
            
            f.write("\n内存使用趋势:\n")
            for usage in profiler.memory_usage:
                f.write(f"  {usage['label']}: {usage['rss_mb']:.2f} MB (RSS), {usage['vms_mb']:.2f} MB (VMS)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("算子执行统计:\n")
            f.write("=" * 80 + "\n\n")
            
            for op_name, stats in profiler.operator_stats.items():
                f.write(f"{op_name}:\n")
                for key, value in stats.items():
                    if key != 'recorded_time' and key != 'tasks':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        logger.info(f"结果已写入: {result_path}")
    except Exception as e:
        logger.error(f"写入结果文件失败: {e}")
    profiler.end_phase("write_results")
    
    # 保存性能指标
    profiler.start_phase("save_performance_metrics")
    try:
        metrics_file = profiler.save_results(result_dir)
    except Exception as e:
        logger.error(f"保存性能指标失败: {e}")
        metrics_file = None
    profiler.end_phase("save_performance_metrics")
    
    # 打印摘要
    logger.info("=" * 80)
    logger.info("执行完成摘要:")
    logger.info("=" * 80)
    logger.info(f"训练数据行数: {training_stats.get('row_count', 0)}")
    logger.info(f"测试数据行数: {test_stats.get('row_count', 0)}")
    logger.info(f"合并后总行数: {combined_stats.get('row_count', 0)}")
    logger.info(f"结果文件: {result_path}")
    if metrics_file:
        logger.info(f"性能指标: {metrics_file}")
    logger.info(f"时间线文件: {timeline_file}")
    
    total_time = sum(metric.get('duration_seconds', 0) for metric in profiler.metrics.values())
    logger.info(f"总执行时间: {total_time:.2f} 秒")
    logger.info("=" * 80)
    
    ray.shutdown()
    
    return result_path

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        sys.exit(1)