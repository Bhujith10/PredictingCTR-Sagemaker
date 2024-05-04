"""Feature engineers the abalone dataset."""
import os
import sys

import argparse
import logging
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import resample

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class Preprocessor:
    def __init__(self, df):
        self.df = df
        self.columns_to_encode = None
        
    def __call__(self, columns_to_encode, encoding_type, train_test_split):
        
        self.columns_to_encode = columns_to_encode.copy()
        self.encoding_type = encoding_type
        self.train_test_split = train_test_split
        
        self.df = self.df.dropna(subset=['session_id','DateTime','user_id','is_click'])
        
        self.df = self.df.drop_duplicates(subset=['DateTime', 'user_id', 'product', 'campaign_id',
       'webpage_id', 'product_category_1', 'product_category_2',
       'user_group_id', 'gender', 'age_level', 'user_depth',
       'city_development_index', 'is_click'])
        
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df['hour'] = self.df['DateTime'].dt.hour
        
        self.df['time_of_day'] = self.df['hour'].apply(self.find_time_of_day)
        
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].fillna(self.df[col].mode().values[0])
            
        for col in self.df.select_dtypes(include=['int','float']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        if self.encoding_type!='target':
            self.df['gender_encoding'] = self.df['gender'].replace({'Female':0,'Male':1})
            self.columns_to_encode.remove('gender')
            
        if self.train_test_split==True:
            self.train = self.df[self.df['DateTime']<'2017-07-06']
            self.validation = self.df[(self.df['DateTime']>='2017-07-06')&(self.df['DateTime']<'2017-07-07')]
            self.test = self.df[(self.df['DateTime']>='2017-07-07')]

            self.encoding()

            return self.train,self.validation,self.test
        
        else:
            self.encoding()

            return self.df
            
    def encoding(self):
        if self.encoding_type=='label':
            for col in self.columns_to_encode:
                le = LabelEncoder()
                if self.train_test_split == True:
                    self.train[f'{col}_encoding'] = le.fit_transform(self.train[col])
                    self.validation[f'{col}_encoding'] = le.transform(self.validation[col])
                    self.test[f'{col}_encoding'] = le.transform(self.test[col])
                else:
                    self.df[f'{col}_encoding'] = le.fit_transform(self.df[col])
                
        if self.encoding_type=='target':
            for col in self.columns_to_encode:
                te = TargetEncoder()
                if self.train_test_split == True:
                    self.train[f'{col}_encoding'] = te.fit_transform(self.train[[col]],self.train[['is_click']])
                    self.validation[f'{col}_encoding'] = te.transform(self.validation[[col]])
                    self.test[f'{col}_encoding'] = te.transform(self.test[[col]])
                else:
                    self.df[f'{col}_encoding'] = te.fit_transform(self.df[col])
                
        if self.encoding_type=='custom':
            for col in self.columns_to_encode:
                if self.train_test_split==True:
                    mapping = self.find_ordinal_mapping(col,self.train)
                    self.train[f'{col}_encoding'] = self.train[col].replace(mapping)
                    self.validation[f'{col}_encoding'] = self.validation[col].replace(mapping)
                    self.test[f'{col}_encoding'] = self.test[col].replace(mapping)
                else:
                    mapping = self.find_ordinal_mapping(col,self.df)
                    self.df[f'{col}_encoding'] = self.df[col].replace(mapping)

    def find_ordinal_mapping(self,col,tmp_df):
        temp_df = tmp_df.groupby([col],as_index=False).agg(no_of_sessions=('session_id','count'),
                                                no_of_clicks=('is_click','sum'))
        temp_df['ctr'] = temp_df['no_of_clicks']/temp_df['no_of_sessions']
        temp_df = temp_df.sort_values('ctr')
        temp_df = temp_df.reset_index().rename(columns={'index':f'{col}_encoding'})
        temp_mapping = dict(zip(temp_df[col],temp_df[f'{col}_encoding']))
        return temp_mapping
        
    def find_time_of_day(self,hour):
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
        
def resample_training_dataset(df, target='is_click', resampling_type='downsample'):
    '''
    Receives the dataset and the sampling technique
    
    Performs the sampling and returns the balanced dataset
    '''
    majority_class = df[df[target] == 0]
    minority_class = df[df[target] == 1]

    if resampling_type=='downsample':
        tmp = resample(majority_class,
                       replace=False,  
                       n_samples=len(minority_class),  
                       random_state=42) 
        
        resampled_df = pd.concat([tmp, minority_class],axis=0)
        
    elif resampling_type=='upsample':
        tmp = resample(minority_class,
                       replace=True,
                       n_samples=len(majority_class),  
                       random_state=42) 
        
        resampled_df = pd.concat([tmp, majority_class],axis=0)

    resampled_df = resampled_df.sample(frac=1, random_state=42)

    return resampled_df

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/predicting_ctr.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading the downloaded csv file")
    df = pd.read_csv(fn)
    
    ## Deletes the file from the location
    os.unlink(fn)
    
    preprocessor = Preprocessor(df)
    columns_to_encode = ['product','gender','age_level','time_of_day','product_category_1','webpage_id','campaign_id']
    encoding_type = 'label'
    train_test_split = True

    train, validation, test = preprocessor(columns_to_encode, encoding_type, train_test_split)
    
    features_to_consider = ['is_click','gender_encoding', 'product_category_1_encoding','product_encoding', 
                        'campaign_id_encoding', 'age_level_encoding','webpage_id_encoding', 
                        'time_of_day_encoding']

    training_dataset = train[features_to_consider]
    validation_dataset = validation[features_to_consider]
    testing_dataset = test[features_to_consider]

    training_dataset = resample_training_dataset(training_dataset)

    logger.info(f"Writing out datasets to {base_dir}")
    pd.DataFrame(training_dataset).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation_dataset).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(testing_dataset).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
