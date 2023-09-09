# Importing the information from the previous component
# finding out the indexes of status of underfit and remove all the contents
# find out the index with the best score 
# Get the best names and its respective parameters
# import the file and create a pickle out of it

from housing_project.entity.components_entity import model_selection_entity
import pickle
import numpy as np
from datetime import datetime
import shutil
import os,sys
from utility.util import *

from sklearn.pipeline import Pipeline
from housing_project.exception import HousingException
from housing_project.logger import logging

path = os.path.join(root_dir, 'Model_container\08_09_23_33_22_\RandomForestRegressor.pkl')

data = model_selection_entity(model_name=['RandomForestRegressor', 'GradientBoostingRegressor', 'SVR'],
 respective_parameters=[{'max_depth': 10, 'n_estimators': 200}, {'learning_rate': 0.3}, {'kernel': 'linear'}], 
 score_training=[0.7074716089658655, 0.7038579165732983, 0.011173781146305073], 
 fitting_status=['Overfitted', 'Overfitted', 'Proper Fit'], 
 module_address=['a','b','c'], 
 pickle_storage_loc=[ path,path,path])
class model_pusher:
    def __init__(self, output_list,transformed_obj_location , BEST_MODEL_DIR,Z_score_loc):
        self.output_list = output_list
        self.BEST_MODEL_DIR = BEST_MODEL_DIR
        self.transformed_obj_location = transformed_obj_location
        self.Z_score_loc = Z_score_loc
    def get_best_model(self) :

        try:

            logging.info('Selecting Best Model')


            # getting the score training from the output list
            score_training = self.output_list[2]
            best_model_loc = (score_training.index(max (score_training)))
            best_pickle_file_loc = self.output_list[5][best_model_loc]
            best_model_score = self.output_list[2][best_model_loc]
            best_model_name = self.output_list[0][best_model_loc]
            best_model_param= self.output_list[1][best_model_loc]

            # saving all of the data to a dictionary

            model_info_to_yaml = { f'Model{datetime.now().strftime("%d_%m_%H_%M_%S_")}':
            { 'Model_name':best_model_name,
                'Model_score':float(best_model_score),
                'Pickle_Location':best_pickle_file_loc,
                'best_model_parameters':best_model_param}
            }

            logging.info(f'Best Model found\n {model_info_to_yaml}')

            

            print(model_info_to_yaml)

            # Sending the above dictionary to the yaml file

            model_entry_to_yaml(model_info_to_yaml)

            # selecting the best model_from model_staus.yaml

            best_model_info = get_best_model_from_yaml()

            # getting the pickle file from the best model:

            best_pickle_file_loc = best_model_info['Pickle_Location']

            


            print(best_pickle_file_loc)

            # saving the file to the best model_location
            with open(best_pickle_file_loc , 'rb') as f:
                model = pickle.load(f)

            # Deleting all the models which are present in the best_model_folder



            Best_model_dir_loc = os.path.join(root_dir,'Best_model')

            shutil.rmtree(Best_model_dir_loc)

            # transfering this final pickle file to the best_model_directory


            current_date_time = datetime.now().strftime("%d_%m_%H_%M_%S_")

            directory = os.path.join(root_dir , self.BEST_MODEL_DIR,current_date_time)
            os.makedirs(directory, exist_ok = True)
            
            # moving this file to the above directory

            shutil.copy(best_pickle_file_loc,directory)
            print(f'Directory for best pickle_file {directory}')



            
            # Now  open the transformed object

    
            with  open(self.transformed_obj_location, 'rb') as f:
                trasnformed_obj = pickle.load(f)

            
            # Combing all three pickle files  into a single pipeline and saving it to a new directory 

            logging.info('Creating a final pipeline and its pickle file which includes steps from Data Transformation and Best Model Selection')

            final_pipeline = Pipeline([
                ('transformation', trasnformed_obj),
                ('final_model', model)

            ])

            # Creating an object of the final pipeline and saving it to the desired location

            final_pipeline_dir_location = os.path.join(root_dir , 'Final_Pipeline')
            os.makedirs(final_pipeline_dir_location , exist_ok=True)
            final_pipeline_obj_location = os.path.join(final_pipeline_dir_location, 'final_pipeline.pkl')

            # Removing everything that is present in the Final_pipeline_folder

            #shutil.rmtree(final_pipeline_dir_location)

            # Creating pipeline and saving it to the Final_pipeline_folder

            with open(final_pipeline_obj_location, 'wb') as f:
                pickle.dump(final_pipeline, f)

        except Exception as e:
            raise HousingException(e, sys) from e

        logging.info(f'Pickle of Final Pipeline created successfully\n loc: {final_pipeline_obj_location}')



        # loading all files to a newer directory with the name Pipeline models and transformers 

        path_to_Pipeline_models_transformers_dir = os.path.join(root_dir , 'Pipeline Models and Transformers')

        os.makedirs(path_to_Pipeline_models_transformers_dir,exist_ok=True)

        # removing everything that is present in the path_to_pipeline_directory and moving all the above pickle files

        #shutil.rmtree(path_to_Pipeline_models_transformers_dir)

        path_z_score = os.path.join(path_to_Pipeline_models_transformers_dir , 'z_score_scaling.pkl' )

        shutil.copy(self.Z_score_loc, path_z_score)

        path_pca_transformer = os.path.join(path_to_Pipeline_models_transformers_dir , 'pca.pkl' )
        shutil.copy(self.transformed_obj_location , path_pca_transformer)

        best_model_path = os.path.join(path_to_Pipeline_models_transformers_dir , 'Best_model.pkl' )

        shutil.copy(best_pickle_file_loc , best_model_path)



        try:

        # Loading the pickle file for the output

            if( final_pipeline_obj_location):
                status = 'Pipeline Completed Successfully'
                logging.info(status)
            
            else:
                status = "Unable to Run Pipeline"
                logging.info(status)

        except Exception as e:
            raise HousingException(e, sys) from e
            
        return status 

         

 

'''

obj = model_pusher(data ,'c:\\Users\\Lokesh\\Desktop\\California Housing Price Prediction 2020\\transformed_obj\\08_09_23_33_00_\\transformation.pkl', 
'Best_Model','C:\\Users\\Lokesh\\Desktop\\California Housing Price Prediction 2020\\Z_Score_Obj\Z_score.pkl')  

output = obj.get_best_model()
 

 
print(f'Output of the data_model is : {output}')
'''

 
 




 



        
        
        # once getting this in the list format find out the location of the max value

         # once find out the max location location get the pickle file 
        # move it to the location of the best location
         
         
                
                
            

 


 # Print the dynamically allocated variables
 