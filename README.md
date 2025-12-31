# 题目J 分布式机器学习预处理算子对比

```text
.
├── assets                            # 图片等资源
├── code                              # 代码文件
│   ├── deploy                        # 部署相关
│   │   ├── images                    # 自定义镜像
│   │   │   └── spark                 # Spark自定义镜像
│   │   ├── infra                     # 基础设施
│   │   │   ├── assets                # 相关配置资源
│   │   │   ├── data                  # 映射数据
│   │   │   └── docker-compose.yaml   # 基础设置Compose配置
│   │   ├── ray                       # Ray集群
│   │   │   └── docker-compose.yml    # Ray集群Compose配置
│   │   └── spark                     # Spark集群
│   │       ├── assets                # Spark相关配置资源（配置文件、脚本、外部JAR等）
│   │       └── docker-compose.yml    # Spark集群Compose配置
│   └── experiments                   # 实验代码
│       ├── ray                       # Ray实验代码
│       │   ├── Exp1
│       │   ├── Exp2
│       │   └── Exp3
│       └── spark                     # Spark实验代码
│           ├── Exp1
│           ├── Exp2
│           └── Exp3
└── README.md                         # 项目说明
```

组号：4

成员：51285903065 郭东旭、52295903013 朱益辉（博士）、51285903055 张京斌、51285903015 杜茗天

## 研究目的

比较Ray Data和Spark MLlib中的机器学习预处理算子。

## 研究内容

通过对比分析Ray Data与Spark MLlib在机器学习数据预处理（如标准化、归一化、特征编码等）算子上的实现机制与性能差异，探究Ray和Spark的差异。

**增加部分**：进一步分析在较大规模数据集（~12GB）下，Ray Data与Spark MLlib在多种数据预处理任务中的扩展性和资源利用效率，评估其在实际大数据处理场景中的适用性。

利用Spark的Eventlog、DAG及PrometheusServlet上报的数据，以及Ray的Dashboard、timeline及上报到Prometheus的CPU、内存利用率数据，分析两者在资源利用率、任务执行时间等方面的差异。

算子选择：

| 实验序号 | 预处理算子    |实验动机 | Spark MLlib                                                                 | Ray Data                                                                                           |
|---|---|---|---|---|
| 1        | 标准化         |全局统计量依赖| `pyspark.ml.feature.StandardScaler`                                         | `ray.data.preprocessors.StandardScaler`                                                            |
| 2        | 特征编码       |高维稀疏数据处理| `pyspark.ml.feature.CountVectorizer`                                        | `ray.data.preprocessors.CountVectorizer`                                                           |
| 3        | 分位数         |近似计算与分桶| `pyspark.sql.DataFrame.approxQuantile`, `pyspark.ml.feature.QuantileDiscretizer` | `ray.data.aggregate.ApproximateQuantile`, `ray.data.preprocessors.CustomKBinsDiscretizer`          |

## 实验

### 实验环境

#### 硬件环境

Ray/Spark集群中包含4个节点，每个节点配置为8核 CPU、32G 内存、32G SHM。

#### 软件环境

使用Docker容器化部署Spark、Ray集群与基础设施，每次实验之间重建集群，确保多次实验之间环境的一致性。

宿主机操作系统 `CentOS 7`, Linux 内核版本 `3.10.0-1160.el7.x86_64`, Docker Engine `26.1.4`, containerd `1.6.33`

Ray/Spark 集群均采用1个主节点 + 3个从节点的架构。

##### Spark 集群

容器基础镜像 `spark:4.0.1-scala2.13-java21-python3-ubuntu`

- JDK 版本： `OpenJDK 21.0.9`
- Spark 版本： `4.0.1 (git revision 29434ea766b)`
- Hadoop 版本： `3.4.1`
- Python 版本： `3.10.12`

Python依赖:

```text
pandas
numpy
scipy
scikit-learn
fsspec
```

额外包含Spark History Server 用于查看 Spark 作业执行详情。

##### Ray 集群

容器基础镜像 `rayproject/ray:2.51.1-py311-cpu`

- Ray 版本： `2.51.1`
- Python 版本： `3.11`

Python依赖:

```text
pandas
numpy
s3fs
fsspec
boto3
pyarrow
psutil
```

##### 基础设施

S3 存储 MinIO： `minio/minio:RELEASE.2025-09-07T16-13-09Z` / `minio/mc:RELEASE.2025-08-13T08-35-41Z`

存储桶信息：

- `bigdata-dataset` ：存放实验数据集
- `bigdata-ray-logs` ：存放 Ray 日志与实验结果
- `bigdata-spark-logs` ：存放 Spark Eventlog 供 Spark History Server 使用

指标收集 Prometheus： `prom/prometheus:latest`

可视化 Grafana： `grafana/grafana-oss:latest`

### 实验负载

选用数据集如下：

#### Airline on-time Performance Data

主要的数值类型数据集 `Airline on-time Performance Data` 选自 Kaggle，包含从1987年10月至2008年4月的商业航班信息，总记录数为 123534969 （1.2亿）条，数据大小约 12GB，原始数据格式为Shuffle过的CSV。数据集链接：[https://www.kaggle.com/datasets/bulter22/airline-data](https://www.kaggle.com/datasets/bulter22/airline-data)。

为了对比不同预处理算子与不同框架在不同规模数据集上的表现，同时为了**保持数据集原始的分布特性**，我们从该数据集中按`Year`字段进行了分层采样，构建了1% ~ 15%，及5% ~ 95%约25种不同规模的数据子集，每个子集保存到`Parquet`格式，并上传至MinIO存储桶`bigdata-dataset`中，供实验使用。

数据中包含多个数值型特征如日期`Year`、`Month`、`DayofMonth`和具有长尾分布的`ArrTime` (到达时间)等，适合用于测试标准化和分位数等预处理算子。

#### MiniPile

为了测试特征编码算子，我们选用了文本数据集 `MiniPile`，该数据集是从大型文本语料库`The Pile`中抽取的一个小型子集，包含约3.1GB的纯文本数据，适合用于测试文本特征编码的效果。数据集链接：[https://huggingface.co/datasets/JeanKaddour/minipile](https://huggingface.co/datasets/JeanKaddour/minipile)。同样将其上传至MinIO存储桶`bigdata-dataset`中，供实验使用。

### 实验步骤

#### 环境搭建

使用Docker Compose搭建基础设施及Ray/Spark集群环境。

使用MinIO作为S3存储，Prometheus用于监控集群指标。
![MinIO](./assets/minio.png)

![Prometheus](./assets/prom.png)

![Prometheus Metric](./assets/prom-metric.png)

搭建Spark集群，可以看到作业执行的Timeline和DAG信息，并配置Spark History Server以便查看作业执行详情。

![Spark](./assets/spark.png)

![Spark Timeline](./assets/spark-timeline.png)

![Spark Job](./assets/spark-job.png)

<img src="./assets/spark-dag.png" alt="Spark DAG" width="20%" />


搭建Ray集群，可以通过Ray Dashboard查看集群状态和任务执行情况。

![Ray Cluster](./assets/ray-cluster.png)

![Ray Job](./assets/ray-job.png)

![Ray Worker](./assets/ray-worker.png)

![Ray Dataline](./assets/ray-data.png)

#### 数据抽样

使用Spark对`Airline on-time Performance Data`数据集按`Year`字段进行分层采样，生成不同规模的数据子集，并保存为Parquet格式上传至MinIO。

![Airline Sample](./assets/airline-sample.png)

![Airline Sample 1%](./assets/finegrained.png)

![Airline Sample 10%](./assets/airline-sample-10.png)

#### 实验一

实验一的设计基于我们对Spark MLlib和Ray Data的调研，选择了标准化这一常用的预处理算子进行对比。标准化在机器学习中广泛应用于数据预处理阶段，有助于提升模型的收敛速度和性能。通过对比两者在不同规模数据集上的表现，我们可以深入了解它们在处理大规模数据时的效率和资源利用情况。

**标准化算子的实现依赖于对整个数据集的全局统计量（均值和标准差）的计算，这使其成为评估分布式计算框架在处理大规模数据时性能的关键指标。**

实验一中对数据进行了标准化处理，分别使用Spark MLlib的`pyspark.ml.feature.StandardScaler`和Ray Data的`ray.data.preprocessors.StandardScaler`对不同规模的Airline数据集进行了标准化处理，并记录了每次实验的性能指标。

每次提交实验任务时，均通过Prometheus收集CPU和内存利用率数据，通过Spark Eventlog和Ray Dashboard收集任务执行时间和资源使用情况。Spark任务使用PySpark编写，进入Spark master容器中使用`spark-submit`提交；Ray任务使用Ray Python API编写，进入Ray head节点容器中使用`ray job submit`提交。提交任务后通过`tee`命令将标准输出和错误输出重定向到日志文件中，便于后续分析。

![Spark Submit](./assets/spark-submit.png)

![Ray Submit](./assets/ray-submit.png)

收集到的数据除写入本地路径外，会额外写入MinIO存储桶`bigdata-spark-logs`和`bigdata-ray-logs`中，便于后续实验结果的整理和分析。

下面为实验一中收集到的部分性能数据示例：
![Ray Perf](./assets/ray-perf.png)

#### 实验二

实验二选择了特征编码中的计数向量化（Count Vectorization）作为对比对象。计数向量化是文本数据处理中常用的技术，能够将文本转换为数值特征，便于后续的机器学习任务。

**通过对比Spark MLlib和Ray Data在处理文本数据时的性能，我们可以评估它们在高维稀疏数据处理方面的能力**。

在实验二中，我们使用MiniPile数据集，分别采用Spark MLlib的`pyspark.ml.feature.CountVectorizer`和Ray Data的`ray.data.preprocessors.CountVectorizer`对文本数据进行了特征编码处理，并记录了每次实验的性能指标。

#### 实验三

实验三选择了分位数计算和分桶（Quantile Discretization）作为对比对象。分位数计算在数据分析和预处理中具有重要意义，能够帮助理解数据的分布特性，而分桶则是将连续变量转换为离散类别的常用方法。

**通过对比Spark MLlib和Ray Data在近似分位数计算和分桶处理方面的性能，我们可以评估它们在处理大规模数据时的效率和准确性**。

在实验三中，我们使用Airline数据集，分别采用Spark MLlib的`pyspark.sql.DataFrame.approxQuantile`和`pyspark.ml.feature.QuantileDiscretizer`，以及Ray Data的`ray.data.aggregate.ApproximateQuantile`和`ray.data.preprocessors.CustomKBinsDiscretizer`对数据进行了分位数计算和分桶处理，并记录了每次实验的性能指标。

### 实验结果与分析

使用表格和图表直观呈现结果，并解释结果背后的原因。

### 结论

总结研究的主要发现。

### 分工

尽可能详细地写出每个人的具体工作和贡献度，并按贡献度大小进行排序。
