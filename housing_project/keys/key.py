from utility.util import read_config

# Basics 


# data variable contains the information of the entire config.yaml file
data = read_config('config.yaml')

URLKEY = (data['Data Ingestion']['url'])

# Data validation 

valid_data = data['Data validation']
SCHEMA_PATH = valid_data['schema']

# Data transformation

transformed_detail = data['Data Transformation']
TRANSFORMED_DIR = transformed_detail['transformed_dir_name']

# Model Selection

model_selection_details = data['Model Selection']
UNDERFITTING_SCORE = model_selection_details['underfitting_score']
MODEL_CONTAINER_DIR = model_selection_details['model_container_dir']

# Model Pusher

pusher_data_details = data['Model Pusher']
BEST_MODEL_DIR= pusher_data_details['best_model_dir']


 

