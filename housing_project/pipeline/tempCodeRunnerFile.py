

X_testing_data_location  = data_ingestion_values[2]
Y_testing_data_location = data_ingestion_values[3]



artifact_time_stamp_folder_loc = data_ingestion_values[4]

data_validation_object = data_validation(X_training_data_location, Y_training_data_location ,
X_testing_data_location, Y_testing_data_location, artifact_time_stamp_folder_loc,data_validation_entity)

data_validation_output = data_validation_object.initiate_data_validation()
print(data_validation_output)

data_transformation_object = data_tranformation(X_training_data_location, Y_training_data_location,
X_testing_data_location,Y_testing_data_location, 3 , 40 , artifact_time_stamp_folder_loc)

data_transformation_output = data_transformation_object.initiate_data_transformation()
print(data_transformation_output)

#X_train_loc, Y_train_loc , X_test_loc, Y_test_loc,z_threshold, vif_score, artifact_time_stamp_loc

# Creating data transformation 


