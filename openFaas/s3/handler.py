import os
import boto3
import json
from time import time

s3_client = boto3.client('s3')

def handle(event):
    try:
        # Your function logic here
        event_data = json.loads(event)

        input_bucket = event_data['input_bucket']
        object_key = event_data['object_key']
        output_bucket = event_data['output_bucket']
        
        # Replace slashes with underscores in the object key
        sanitized_object_key = object_key.replace('/', '_')


        path = '/tmp/' + sanitized_object_key

        start = time()
        s3_client.download_file(input_bucket, object_key, path)
        download_time = time() - start

        start = time()
        s3_client.upload_file(path, output_bucket, object_key)
        upload_time = time() - start

        os.remove(path)  # Remove the downloaded file after upload

        return {"download_time": download_time, "upload_time": upload_time}
    except Exception as e:
        print(f"Error during function execution: {str(e)}")
        raise
