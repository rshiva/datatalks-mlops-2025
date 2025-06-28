import pandas as pd
from datetime import datetime
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

input_file = 's3://nyc-duration/in/2023-01.parquet'
output_file = 's3://nyc-duration/out/2023-01.parquet'

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system("python batch.py 2023 1")

df_result = pd.read_parquet(output_file, storage_options=options)

print(f"The sum of predicted durations is: {df_result['predicted_duration'].sum()}")