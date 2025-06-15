
import pickle
import pandas as pd
import sys
import os


def read_data(filename ,year,month):
    df = pd.read_parquet(filename,engine='pyarrow')
    
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def save_results():

    with open('lin_reg.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # March 03
    # taxi_type = sys.argv[3] # green or yellow


    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet',year,month)
    

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("mean predictions :",y_pred.mean())

    # output_file = f's3://prediction-manthan/taxi_type=yellow/year={year:04d}/month={month:02d}.parquet'

    output_file = f'./data/output/yellow/year={year:04d}/month={month:02d}.parquet'

    df_result = pd.DataFrame()
    df_result['predictions'] = y_pred
    df_result['ride_id'] = df['ride_id']

    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )


if __name__ == '__main__':
    save_results() 