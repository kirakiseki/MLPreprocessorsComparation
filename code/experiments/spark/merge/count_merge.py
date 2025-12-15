#!/usr/bin/env python3
"""
统计合并两个CSV文件的行数
"""

from pyspark.sql import SparkSession
import os
import sys
import time

def main():
    # 创建SparkSession
    spark = SparkSession.builder \
        .appName("CountMergeCreditData") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("INFO")
    spark.catalog.clearCache()
    
    print("=" * 60)
    print("开始读取CSV文件...")
    print("=" * 60)
    
    # 定义文件路径
    training_path = "s3a://bigdata-dataset/GiveMeSomeCredit/cs-training.csv"
    test_path = "s3a://bigdata-dataset/GiveMeSomeCredit/cs-test.csv"
    file_path = os.path.dirname(__file__)
    result_dir = os.path.join(file_path, "run", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "result.txt")
    
    try:
        # 读取训练数据
        print(f"正在读取训练数据: {training_path}")
        df_training = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(training_path)
        
        # 读取测试数据
        print(f"正在读取测试数据: {test_path}")
        df_test = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(test_path)
        
        # 显示数据信息
        print("\n训练数据集信息:")
        print(f"训练数据行数: {df_training.count()}")
        print(f"训练数据列数: {len(df_training.columns)}")
        print("训练数据schema:")
        df_training.printSchema()
        
        print("\n测试数据集信息:")
        print(f"测试数据行数: {df_test.count()}")
        print(f"测试数据列数: {len(df_test.columns)}")
        print("测试数据schema:")
        df_test.printSchema()
        
        # 合并数据
        print("\n正在合并数据集...")
        # 确保两个DataFrame有相同的schema
        # 由于测试数据可能缺少目标列，我们可以使用unionByName并设置allowMissingColumns
        try:
            df_combined = df_training.unionByName(df_test, allowMissingColumns=True)
        except:
            # 如果列名不完全匹配，使用union
            print("注意：两个数据集的列不完全相同，使用union...")
            df_combined = df_training.union(df_test)
        
        # 统计合并后的行数
        total_count = df_combined.count()
        
        print("=" * 60)
        print("统计结果:")
        print("=" * 60)
        print(f"训练数据集行数: {df_training.count()}")
        print(f"测试数据集行数: {df_test.count()}")
        print(f"合并后总行数: {total_count}")
        print("=" * 60)
        
        # 保存统计结果到本地文件（可选）
        with open(result_path, 'w') as f:
            f.write(f"训练数据集行数: {df_training.count()}\n")
            f.write(f"测试数据集行数: {df_test.count()}\n")
            f.write(f"合并后总行数: {total_count}\n")
        
        print(f"结果已保存到 {result_path}")
        
        # 显示前5行数据（可选）
        print("\n合并数据前5行:")
        df_combined.show(5)
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 停止SparkSession
        spark.stop()
        print("\nSpark作业执行完成！")

if __name__ == "__main__":
    main()