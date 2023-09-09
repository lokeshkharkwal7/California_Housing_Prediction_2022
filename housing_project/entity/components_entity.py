from collections import namedtuple

# Data ingestion entity 

data_ingestion_entity = namedtuple("data_ingestion_entity",['train_input_loc', 'train_target_loc',
'test_input_loc', 'test_target_loc','artifact_time_stamp_loc'])

# Data validation entity 

data_validation_entity = namedtuple("data_validation_entity", ['is_validated', 'Data_drift_file_location'])

# Data Transformation entity 

data_tranformation_entity = namedtuple("data_transformed_entity", ['transformed_x_train_loc' ,'y_train_loc', 'tranformed_x_test_loc',
 'y_test_loc','transformed_obj_loc','Z_score_loc'])

 # Model Selection entitu

model_selection_entity = namedtuple("model_selection_entity", ['model_name','respective_parameters',
 'score_training','fitting_status','module_address','pickle_storage_loc'])