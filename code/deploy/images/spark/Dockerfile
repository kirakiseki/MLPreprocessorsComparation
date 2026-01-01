FROM spark:4.0.1-scala2.13-java21-python3-ubuntu

USER root

COPY requirements.txt /opt/spark/requirements.txt

RUN pip install -r /opt/spark/requirements.txt --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple