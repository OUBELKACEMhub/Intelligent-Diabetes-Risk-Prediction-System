FROM apache/airflow:3.1.7-python3.12

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /app

COPY . /app


USER root

RUN mkdir -p /var/log/supervisor \
    && mkdir -p /app/mlartifacts \
    && chown -R airflow:root /app /var/log/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

USER airflow

EXPOSE 8000 8501 5000 8080

ENTRYPOINT ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]