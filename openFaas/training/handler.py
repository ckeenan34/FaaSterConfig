# handler.py

import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import re
import io
import sys
import json
from time import time

s3_client = boto3.client('s3')
cleanup_re = re.compile('[^a-z]+')
tmp = '/tmp/'


def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence


def train_model(dataset_bucket, dataset_object_key, model_bucket, model_object_key):
    obj = s3_client.get_object(Bucket=dataset_bucket, Key=dataset_object_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    df['train'] = df['Text'].apply(cleanup)

    tfidf_vector = TfidfVectorizer(min_df=100).fit(df['train'])

    train = tfidf_vector.transform(df['train'])

    model = LogisticRegression()
    model.fit(train, df['Score'])

    model_file_path = tmp + model_object_key
    joblib.dump(model, model_file_path)

    s3_client.upload_file(model_file_path, model_bucket, model_object_key)


def handler():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())

        dataset_bucket = input_data['dataset_bucket']
        dataset_object_key = input_data['dataset_object_key']
        model_bucket = input_data['model_bucket']
        model_object_key = input_data['model_object_key']

        start = time()
        train_model(dataset_bucket, dataset_object_key, model_bucket, model_object_key)
        latency = time() - start

        # Write output to stdout
        output_data = {'latency': latency}
        print(json.dumps(output_data))

    except Exception as e:
        print(f"Error: {str(e)}")


# Uncomment the line below for local testing
# handler()
