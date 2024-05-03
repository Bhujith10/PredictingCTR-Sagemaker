import json
import pandas as pd
import boto3
import os


ENDPOINT_NAME= os.environ['ENDPOINT_NAME']

gender_encoding = {'female':0,'male':1}
product_category_1_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
product_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
campaign_mapping = {82320: 0, 98970: 1, 105960: 2, 118601: 3, 359520: 4, 360936: 5, 396664: 6, 404347: 7, 405490: 8, 414149: 9}
age_level_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 6.0: 6}
webpage_mapping = {1734: 0, 6970: 1, 11085: 2, 13787: 3, 28529: 4, 45962: 5, 51181: 6, 53587: 7, 60305: 8}
time_of_day_mapping = {'afternoon': 0, 'early-morning': 1, 'evening': 2, 'morning': 3, 'night': 4}

    
def fetch_from_feature_store(user_id):
    '''
    Fetches the most recent clicked product by the user from customers feature stor
    '''
    featurestore_runtime_client = boto3.client('sagemaker-featurestore-runtime',region_name='us-east-1')
    feature_record = featurestore_runtime_client.get_record(FeatureGroupName='feature-store-predicting-ctr-customers-04-28-17-06', 
                                                        RecordIdentifierValueAsString=str(user_id))
    most_clicked_product = -1
    if 'Record' in feature_record:
        for record in feature_record['Record']:
            if record['FeatureName'] == 'most_recent_clicked_product_encoding':
                most_clicked_product = float(record['ValueAsString'])
                break
            
    return most_clicked_product
       
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

def lambda_handler(event, context):
    temp_df = pd.DataFrame(event['data'])
    user_id = temp_df[0].values[0]
    temp_df = temp_df.drop(columns=temp_df.columns[0],axis=1)
    temp_df[1] = temp_df[1].apply(lambda x:x.lower()).replace(gender_encoding)
    temp_df[2] = temp_df[2].replace(product_category_1_mapping)
    temp_df[3] = temp_df[3].replace(product_mapping)
    temp_df[4] = temp_df[4].replace(campaign_mapping)
    temp_df[5] = temp_df[5].replace(age_level_mapping)
    temp_df[6] = temp_df[6].replace(webpage_mapping)
    temp_df[7] = temp_df[7].apply(find_time_of_day)
    temp_df[7] = temp_df[7].replace(time_of_day_mapping)
    temp_df[8] = fetch_from_feature_store(user_id)
    
    
    sagemaker_runtime = boto3.client('sagemaker-runtime',region_name='us-east-1')
    
    prediction_results = []
    
    for i in range(len(temp_df)):

        serialized_input = ','.join(map(str,temp_df.values[i]))
        print(serialized_input)
    
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=serialized_input
        )
    
        result = response["Body"].read().decode("ascii")
        
        probability = json.loads(result)[0]
        
        prediction = 'Clicked' if probability>0.5 else 'Not Clicked'
        
        prediction_results.append(prediction)
    
    return {
        'statusCode': 200,
        'prediction': prediction_results
    }
