# One image for every Python service (producer, spark job, scorer, api, ...).
FROM python:3.11-slim-bookworm

# Spark needs Java
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps curl \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 PYTHONUNBUFFERED=1 PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt requirements-nlp.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_NLP=false
RUN if [ "$INSTALL_NLP" = "true" ]; then \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-nlp.txt; fi

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
