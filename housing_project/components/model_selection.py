# Here creating a models.yaml file which contains the imformation of all of all of the 
# models we will use in this project

# Creating a function that will use grid search cv to find out the best parameters for
# all of the models

# Saving the adjusted r2 score of all of the models into a data frame

# Selecting the best and transfering it into a pickle file 

from housing_project.components.data_ingestion_component import data_ingestion_component
from housing_project.components.data_transformation_component import data_tranformation
from housing_project.entity.components_entity import model_selection_entity
from housing_project.logger import logging
from housing_project.exception import HousingException
import pandas as pd
from utility.util import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import importlib
from datetime import datetime

from sklearn.metrics import mean_squared_error
import pickle
import math
import shutil
import os , sys
 




class model_selection:
    def __init__(self, x_train_transformed_loc , y_train_data_loc, x_test_transformed_loc, 
    y_test_data_loc,UNDERFITTING_SCORE,MODEL_CONTAINER_DIR):

        self.x_train_transformed_loc = x_train_transformed_loc
        self.y_train_data_loc = y_train_data_loc
        self.x_test_transformed_loc = x_test_transformed_loc
        self.y_test_data_loc = y_test_data_loc
        self.UNDERFITTING_SCORE = UNDERFITTING_SCORE
        self.MODEL_CONTAINER_DIR = MODEL_CONTAINER_DIR
         
    def initiate_model_selection(self):

        try:

            logging.info('Initiating model Selection')

            # Initiating values:
            model_performance_test = []
            model_performance_final = []
            fitting_status_= []
            score_training_= []
            respective_parameters_= []
            model_name_= []
            module_address=[]
            pickle_storage = []

            # creating a directory to save all the pric files present in the model

            dir = root_dir

            current_date_time = datetime.now().strftime("%d_%m_%H_%M_%S_")

            directory = os.path.join(dir , self.MODEL_CONTAINER_DIR,current_date_time)
            os.makedirs(directory, exist_ok = True)


            # Reading the x_train_transformed, y_train, x_test_transformed, y_test

            x_train_transformed = pd.read_csv(self.x_train_transformed_loc)
            print('info about X_train_transformed')
            print(x_train_transformed.info())
            y_train = pd.read_csv(self.y_train_data_loc)
            print('info about Y_train_transformed')
            print(y_train.info())

            x_test_transformed= pd.read_csv(self.x_test_transformed_loc)

            print("Y test info")
            print(x_test_transformed.info())
            y_test= pd.read_csv(self.y_test_data_loc)
            print("Y test info")
            print(y_test.info())

            #  Removing Unnamed: 0 from our transformed data frames

            for i in [x_train_transformed,x_test_transformed,y_test,y_train]:
                i.drop(columns = 'Unnamed: 0', inplace = True)
            
            #  reading config files from model.config

            model_data = read_config('model.yaml')
            model_data_information = model_data['models']

            logging.info(f'Taking all models and there information from the config file\n {model_data_information}')

            

            # Checking underfitting and overfitting

                # splitting the x_train transformed into split_train_x__transformed and  split_train_y_transformed  

            split_x_train,split_x_test , split_y_train , split_y_test = train_test_split(x_train_transformed,y_train, test_size = 0.33 , random_state = 42)
        
        except Exception as e:
            raise HousingException(e, sys) from e 

         # training the model with the splitting datas
         # Testing the datas with the same model
         # print the model
         # calculate the difference


        for i in model_data_information:

            try:
                model_imfo = i
                model_name = model_imfo['name']
                model_parameteres = model_imfo['parameters']

                logging.info(f'Checking Overfitting for {model_name}')

                # dynamically importing the model_name 
                library = importlib.import_module("sklearn.ensemble" if model_name in ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"] else "sklearn.svm")
                model_class = getattr(library, model_name)
                final_model = model_class()


                # using grid_search_cv to find out the best parameters
                # Creating an object of gridsearchcv

                grid_search_obj = GridSearchCV(final_model,model_parameteres)
                # Fitting this object to our train and test data 
                grid_search_obj.fit(split_x_train,split_y_train)
                best_parameters = grid_search_obj.best_params_
                y_pred_test = grid_search_obj.predict(split_x_test)

                y_pred_final = grid_search_obj.predict(x_test_transformed)


                score_training = r2_score(split_y_test, y_pred_test,force_finite=False)
                score_testing = r2_score(y_test ,y_pred_final,force_finite=False)
                fitting_differnce = abs(score_testing)-abs(score_training)
                if abs(fitting_differnce)<self.UNDERFITTING_SCORE:
                    fitting_status = 'Proper Fit'
                else:
                    fitting_status = 'Overfitted'
                logging.info(f'Fitting status {fitting_status}')
                logging.info(f'Model_score {score_training}')

                squared_error = mean_squared_error(y_pred_final, y_test)
                squared_error=math.sqrt(squared_error)
                squared_error_z = mean_squared_error(split_y_test, y_pred_test)

                squared_error_z=math.sqrt(squared_error)

                # abs will convert any negative value to positive value
                diff = abs(squared_error_z)-abs(squared_error)
                file_name = f'{model_name}.pkl'



    
                

                with open(file_name,'wb') as f:
                    pickle.dump(grid_search_obj,f)

                # moving the pickel file to model container

                logging.info('Saving file to the Model Container directory')
                shutil.move(file_name , directory) 
                pickle_location = os.path.join(directory, file_name)
            

                pickle_storage.append(pickle_location)



    
                # Appending the useful information to the respective list
                
                model_name_.append(model_name)
                respective_parameters_.append(best_parameters)
                score_training_.append(score_training)

                model_performance_test.append(score_testing)

                fitting_status_.append(fitting_status)
                module_address.append(model_class)

                print(f'Testing Score for the models are {score_training_}')


                print(f'Testing Score for the models are {model_performance_test}')

            except Exception as e:
                raise HousingException(e, sys) from e
            
            

                

        model_selection_ouput = model_selection_entity(model_name=model_name_ , respective_parameters=respective_parameters_,
        score_training=score_training_,fitting_status=fitting_status_,module_address=module_address,pickle_storage_loc = pickle_storage)

        logging.info(f'Model Selection Completed \n Information: {model_selection_ouput}')


        

        return model_selection_ouput

        


# create a list of fitting_difference 
# create a list of model_name and score testing 
# filter the location which does not have underfitting
# find the location of the testing score of the above filterd value which have the highest value
# now take that location and print the name of the model 
# return the name tuple
            

'''            
 
            # saving the r2 score and name of the file into a dictionary
transformed_x_train_loc='artifact\\\\27_08_15_41_08_\\\\Transformed_data\\\\train_transformed_data\\\\transf_X_train.csv'
y_train_loc='artifact\\27_08_15_41_08_\\Transformed_data\\train_transformed_data\\transf_target_train.csv'
tranformed_x_test_loc='artifact\\\\27_08_15_41_08_\\\\Transformed_data\\\\test_transformed_data\\\\transf_X_test.csv'
y_test_loc='artifact\\27_08_15_41_08_\\Transformed_data\\train_transformed_data\\transf_target_test.csv'
obj = model_selection(transformed_x_train_loc,y_train_loc,tranformed_x_test_loc,y_test_loc,0.25,'Model_container')
output = obj.initiate_model_selection()
print(output) 
'''

 
 