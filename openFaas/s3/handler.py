import os
import boto3
from time import time

s3_client = boto3.client('s3')

def s3du(event, context):
    print("hello")
    try:
        # Your function logic here

        input_bucket = event['input_bucket']
        object_key = event['object_key']
        output_bucket = event['output_bucket']
        print(input_bucket)
        print(object_key)
        print(output_bucket)

        path = '/tmp/' + object_key

        start = time()
        s3_client.download_file(input_bucket, object_key, path)
        download_time = time() - start

        start = time()
        s3_client.upload_file(path, output_bucket, object_key)
        upload_time = time() - start

        os.remove(path)  # Remove the downloaded file after upload

        return {"download_time": download_time, "upload_time": upload_time}
    except Exception as e:
        logging.error(f"Error during function execution: {str(e)}")
        raise
