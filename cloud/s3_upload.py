"""
Optional: copy the local data lake (bronze Parquet + daily prices) to AWS S3,
so it can be queried by Athena, Databricks or loaded into Snowflake (cloud/snowflake_load.sql).
Needs AWS credentials (aws configure) and AWS_S3_BUCKET in .env.
"""
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()
bucket = os.environ["AWS_S3_BUCKET"]
s3 = boto3.client("s3")
count = 0
for folder in ["data/bronze", "data/daily"]:
    for f in Path(folder).rglob("*.parquet"):
        s3.upload_file(str(f), bucket, f"stock-pipeline/{f.as_posix()}")
        count += 1
print(f"uploaded {count} files to s3://{bucket}/stock-pipeline/")
