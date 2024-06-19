import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date
import logging

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

sagemaker_session = sagemaker.Session()

def modify_data_in_feature_store_format(df, feature_group_name):
    tmp_client = boto3.client('sagemaker')
    response = tmp_client.describe_feature_group(FeatureGroupName=feature_group_name)
    transaction_features = [i['FeatureName'] for i in response['FeatureDefinitions'] if i['FeatureName']!='event_time']
    transaction_features_datatypes = {i['FeatureName']:i['FeatureType'] for i in response['FeatureDefinitions'] if i['FeatureName']!='event_time'}
    
    df = df[transaction_features]
    
    for feature,datatype in transaction_features_datatypes.items():
        if datatype=='Integral':
            df[feature] = df[feature].astype('int')
        elif datatype=='String':
            df[feature] = df[feature].astype('string')
        elif datatype=='Fractional':
            df[feature] = df[feature].astype('float')
        else:
            df[feature] = df[feature].astype('string')
    
    df['event_time'] = generate_event_timestamp()
    df['event_time'] = df['event_time'].astype('string')
    
    return df


def ingest_data_into_feature_store(df, feature_group_name):
    logger.info(df.info())
    logger.info(f"Ingesting data into {feature_group_name} Feature Store")
    sagemaker_session = sagemaker.Session()
    session = boto3.session.Session()
    featurestore_runtime_client = session.client(service_name='sagemaker-featurestore-runtime')
    fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    response = fg.ingest(data_frame=df, max_processes=32, wait=True)
    """
    The ingest call above returns an IngestionManagerPandas instance as a response. Zero based indices of rows 
    that failed to be ingested are captured via failed_rows in this response. By confirming this count to be,
    we validated that all rows were successfully ingested without a failure.
    """
    assert len(response.failed_rows) == 0

def find_most_clicked_product(group):
    rolling_window = group.rolling(window=24)
    session_ids = []
    products = []
    for win in rolling_window:
        session_ids.append(win.iloc[-1]['session_id'])
        if 1 in win['is_click']:
            products.append(win.groupby(['product'],as_index=False)['is_click'].sum().nlargest(1,'is_click')['product'].values[0])
        else:
            products.append(win['product'].value_counts().nlargest(1).index[0])
    return pd.DataFrame({
        'session_id':session_ids,
        'most_recent_clicked_product':products
                 })
    
def find_most_clicked_webpage(group):
    rolling_window = group.rolling(window=24)
    session_ids = []
    webpage_ids = []
    for win in rolling_window:
        session_ids.append(win.iloc[-1]['session_id'])
        if 1 in win['is_click']:
            webpage_ids.append(win.groupby(['webpage_id'],as_index=False)['is_click'].sum().nlargest(1,'is_click')['webpage_id'].values[0])
        else:
            webpage_ids.append(win['webpage_id'].value_counts().nlargest(1).index[0])
    return pd.DataFrame({
        'session_id':session_ids,
        'most_recent_clicked_webpage':webpage_ids
                 })

def find_time_of_day(hour):
    if hour>0 and hour<5 or hour>22:
        return 'night'
    elif hour>=5 and hour<10:
        return 'early-morning'
    elif hour>=10 and hour<=13:
        return 'morning'
    elif hour>13 and hour<18:
        return 'afternoon'
    else:
        return 'evening'
        
def generate_event_timestamp():
    # naive datetime representing local time
    naive_dt = datetime.now()
    # take timezone into account
    aware_dt = naive_dt.astimezone()
    # time in UTC
    utc_dt = aware_dt.astimezone(timezone.utc)
    # transform to ISO-8601 format
    event_time = utc_dt.isoformat(timespec='milliseconds')
    event_time = event_time.replace('+00:00', 'Z')
    return event_time
        
s3_uri_prefix = 's3://predicting-ctr/sagemaker-feature-store/dataset.csv'
transactions_feature_group_name = "feature-store-predicting-ctr-transactions-04-28-17-06"
customers_feature_group_name = "feature-store-predicting-ctr-customers-04-28-17-06"

logger.info("started operations")
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='predicting-ctr', Key='sagemaker-feature-store/dataset.csv')
df = pd.read_csv(obj['Body'])

df.head()

columns_to_encode = ['product','gender','age_level','time_of_day','product_category_1','webpage_id','campaign_id']
df = df.dropna(subset=['session_id','DateTime','user_id','is_click'])
        
df = df.drop_duplicates(subset=['DateTime', 'user_id', 'product', 'campaign_id',
'webpage_id', 'product_category_1', 'product_category_2',
'user_group_id', 'gender', 'age_level', 'user_depth',
'city_development_index', 'is_click'])

df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour

df['time_of_day'] = df['hour'].apply(find_time_of_day)

df.sort_values(['user_id','DateTime'],inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode().values[0])

for col in df.select_dtypes(include=['int','float']).columns:
    df[col] = df[col].fillna(df[col].median())
    
temp_results = df.groupby(['user_id']).apply(find_most_clicked_product)
temp_results = temp_results.reset_index()
df = df.merge(temp_results[['user_id','session_id','most_recent_clicked_product']],on=['user_id','session_id'],how='left')

temp_results = df.groupby(['user_id']).apply(find_most_clicked_webpage)
temp_results = temp_results.reset_index()
df = df.merge(temp_results[['user_id','session_id','most_recent_clicked_webpage']],on=['user_id','session_id'],how='left')

df['gender_encoding'] = df['gender'].replace({'Female':0,'Male':1})
columns_to_encode.remove('gender')

encoder_dictionary = {}
        
for col in columns_to_encode:
    le = LabelEncoder()
    df[f'{col}_encoding'] = le.fit_transform(df[col])
    encoder_dictionary[col] = {label: encoded_value for label, encoded_value in zip(le.classes_, le.transform(le.classes_))}
    
df['most_recent_clicked_product_encoding'] = df['most_recent_clicked_product'].replace(encoder_dictionary['product'])
df['most_recent_clicked_webpage_encoding'] = df['most_recent_clicked_webpage'].replace(encoder_dictionary['webpage_id'])

customers_df = df[['user_id','gender_encoding','age_level_encoding','most_recent_clicked_product_encoding','most_recent_clicked_webpage_encoding']]
customers_df = customers_df.groupby(['user_id'],as_index=False).agg(gender_encoding=('gender_encoding','first'),
                                                    age_level_encoding=('age_level_encoding','first'),
                                                    most_recent_clicked_product_encoding=('most_recent_clicked_product_encoding','max'),
                                                    most_recent_clicked_webpage_encoding=('most_recent_clicked_webpage_encoding','max'))

df = modify_data_in_feature_store_format(df,transactions_feature_group_name)
ingest_data_into_feature_store(df,transactions_feature_group_name)

customers_df = modify_data_in_feature_store_format(customers_df,customers_feature_group_name)
ingest_data_into_feature_store(customers_df,customers_feature_group_name)

job.init(args['JOB_NAME'], args)
job.commit()