from housing_project.components.data_ingestion_component import data_ingestion_component
from housing_project.configuration.housing_configuration import ingestion_configuration
from housing_project.components.data_transformation_component import data_tranformation
from housing_project.configuration.housing_configuration import data_transformation_configuration
from housing_project.components.data_validation_component import data_validation
from housing_project.entity.components_entity import data_validation_entity
from housing_project.components.model_selection import model_selection
from housing_project.configuration.housing_configuration import model_selection_configuration
from housing_project.components.model_pusher import model_pusher
from housing_project.configuration.housing_configuration import model_pusher_configuration
from housing_project.exception import HousingException
from housing_project.keys.key import *
import math
import pandas as pd
import os,sys


class Pipeline:

    def __init__(self):
        pass
    def run_pipeline(self):


        # Initiating data Ingestion

        try:

            data_ingestion_object = data_ingestion_component(URLKEY)
            data_ingestion_values = data_ingestion_object.initiate_data_ingestion()
            

            X_training_data_location = data_ingestion_values[0]
            Y_training_data_location = data_ingestion_values[1]

            X_testing_data_location  = data_ingestion_values[2]
            Y_testing_data_location = data_ingestion_values[3]

            print(data_ingestion_values)



            artifact_time_stamp_folder_loc = data_ingestion_values[4]

            

            # Initiating Data Validation

            data_validation_object = data_validation(X_training_data_location, Y_training_data_location ,
            X_testing_data_location, Y_testing_data_location, artifact_time_stamp_folder_loc,data_validation_entity)

            data_validation_output = data_validation_object.initiate_data_validation()
            
            # Initiating Data Transformation

            data_transformation_object = data_tranformation(X_training_data_location, Y_training_data_location,
            X_testing_data_location,Y_testing_data_location, 3 , 80 , artifact_time_stamp_folder_loc)

            data_transformation_output = data_transformation_object.initiate_data_transformation()

            print(data_transformation_output)


            # Initiating model selection 

            X_training_data_location=data_transformation_output[0]
            Y_training_data_location =data_transformation_output[1]
            X_testing_data_location=data_transformation_output[2]
            Y_testing_data_location=data_transformation_output[3]
            config_obj = model_selection_configuration()
            underfitting_score = config_obj.get_underfitting_score()
            model_container_dir = config_obj.get_model_container_dir()

            model_selection_obj = model_selection(X_training_data_location,Y_training_data_location,X_testing_data_location,
            Y_testing_data_location,underfitting_score,model_container_dir)
            output = model_selection_obj.initiate_model_selection ()
            print(output)


            # Initiating model pusher 
            

            transformed_obj_location  = data_transformation_output[4]
            Z_score_loc  = data_transformation_output[5]

            model_pusher_config = model_pusher_configuration()
            best_model_dir = model_pusher_config.get_best_modeldir()
            print(best_model_dir)
            model_pusher_obj = model_pusher(output , transformed_obj_location, best_model_dir  
            ,Z_score_loc)

            final_pipeline_obj_location = model_pusher_obj.get_best_model()
            print(final_pipeline_obj_location)

            # creating a final component which is called Final_output 

            # In this component combine transformed and best_model_data into a single object 

            # return its location from pipeline so that it can be used for the combie output building

        except Exception as e:
            raise HousingException(e, sys) from e
 
'''
object = Pipeline()
object.run_pipeline()
'''
 
            

 


   


 


