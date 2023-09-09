

from housing_project.components import data_ingestion_component
from housing_project.components import data_validation_component
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os, sys
from housing_project.logger import logging
from utility.util import *
from housing_project.exception import HousingException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer




from housing_project.entity.components_entity import data_tranformation_entity
import numpy as np

import shutil
from pandas_profiling import ProfileReport




class data_tranformation:
    def __init__(self,X_train_loc, Y_train_loc , X_test_loc, Y_test_loc,z_threshold, vif_score, artifact_time_stamp_loc):

        logging.info('Starting Data Transformation')
        self.x_train_loc = X_train_loc
        self.y_train_loc = Y_train_loc
        self.x_test_loc = X_test_loc
        self.y_test_loc = Y_test_loc
        self.z_threshold = z_threshold
        self.vif_score = vif_score
        self.artifact_time_stamp_loc = artifact_time_stamp_loc


    def initiate_data_transformation(self):

        try:

        
            # importing train and test data_sets
            x_train = pd.read_csv(self.x_train_loc)
            y_train = pd.read_csv(self.y_train_loc)
            y_train = y_train['medianHouseValue']
            y_train = pd.DataFrame(y_train, columns = ['medianHouseValue'])

            x_train['medianHouseValue'] = y_train
            x_test = pd.read_csv(self.x_test_loc)
            y_test = pd.read_csv(self.y_test_loc)
            y_test= y_test['medianHouseValue']
            y_test = pd.DataFrame(y_test, columns = ['medianHouseValue'])
            x_test['medianHouseValue'] = y_test


            for i in [x_train, x_test]:
                i.drop(columns='Unnamed: 0', inplace=True)
            
            dir = root_dir


            # Handeling the missing values using simple imputer 

            logging.info('Handeling Null values with Simple Imputer')

            imputer_obj = SimpleImputer(strategy = 'median')
            imputer_obj.fit(x_train)
            x_train_imputed =imputer_obj.transform(x_train)
            imputer_obj.fit(x_test)
            x_test_imputed = imputer_obj.transform(x_test)
        except Exception as e:
            raise HousingException (e, sys) from e

        # checking the outliers present in both training and testing datasets

        #combing data frames again into one single DF so that we can detect and remove outlier

        def outlier_removal(dataframe):

            try:

                logging.info('Null Values successfully handeled now handeling outlier using zscore_threshold')
                z_threshold = self.z_threshold

                    # Calculate Z-Scores for the columns you're interested in or select whole
                z_scores = np.abs(stats.zscore(dataframe))

                    # Create a boolean mask to identify outliers

                    # This will return true and fase with all the values which is smaller than 
                outlier_mask = (z_scores > z_threshold).any(axis=1)

                
                df_no_outliers_train = dataframe[~outlier_mask]
                df_no_outliers_train = pd.DataFrame(df_no_outliers_train, columns = x_train.columns)
                return df_no_outliers_train

            except Exception as e:
                 raise HousingException(e , sys) from e

        #Calling above function and storing the values in df_no_outlier data frame

        try:

            df_no_outliers_train = outlier_removal(x_train_imputed)
            df_no_outliers_test = outlier_removal(x_test_imputed)

            # Redeclearing x_train , x_test , y_train, y_test

            #creating Y
            y_test= df_no_outliers_test['medianHouseValue']
            y_test = pd.DataFrame(y_test, columns = ['medianHouseValue'])

            y_train= df_no_outliers_train['medianHouseValue']
            y_train = pd.DataFrame(y_train, columns = ['medianHouseValue'])

            #creating X
            x_train = df_no_outliers_train.drop(columns = 'medianHouseValue')
            x_test = df_no_outliers_test.drop(columns = 'medianHouseValue')

            # Removing the multicollinearity 
                #scaling the data  


            # Using z score normalisation

            logging.info('Outlier Successfully Handeled started Z score sacalilng of the data')

            scaler = StandardScaler()
            arr = scaler.fit_transform(x_train)


                # using variation_inflation_factor from the statsmodel liberary
            
            logging.info('Handeling Multicollinearity using VIF score')

            vif_data = pd.DataFrame({'Feature': x_train.columns, 'VIF Score': [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]})
            vif_data = pd.DataFrame(vif_data)

                # using variation inflation factor score to remove any value above then the given threshold

            variation_inflation_score = self.vif_score
            removing_columns_df = vif_data[vif_data['VIF Score']>variation_inflation_score]
            r_column_name = removing_columns_df['Feature']

    
            modified_x_train = x_train.drop(columns=r_column_name)
            modified_x_test = x_test.drop(columns=r_column_name)

            print('column that were present are ',modified_x_train.columns)

            logging.info(f'Removed columns with the highest VIF score are : {r_column_name}')
 


            ## Scaling the data to get the object of the standard sclare of the selected columns without the vif score

            scaler2 = StandardScaler()
            scaler2.fit(x_train)
 

            ## Creating a pickle file and moving this obj to a new dir location 


            ########### Remove this if everything is working fine

            z_score_pkl_loc = os.path.join(dir,'Z_Score_Obj')
            os.makedirs(z_score_pkl_loc, exist_ok=True)

            file_location_z_score = os.path.join(z_score_pkl_loc,'Z_score.pkl')


            with open(file_location_z_score, 'wb') as f:
                pickle.dump(scaler,f)

        except Exception as e:
            raise HousingException(e , sys) from e




 # Dimensity reduction of the data using the PCA 

        # Step 1. Scaling the data 

        try:

            logging.info('Performing Dimensity reduction using PCA')

            x_train_scaled = scaler.transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            #  Step 2 Transforming the data into PCA

            pca = PCA()

            x_train_pca_dataframe = pd.DataFrame(pca.fit_transform(x_train_scaled))

            x_test_pca_dataframe = pd.DataFrame(pca.transform(x_test_scaled))

            # ( write a code that will save the graph of the pca to the location)
            
            # Specifing the n components by looking on the pca graph 

            ##############################################################3

            pca1= PCA(n_components=6)

            train_pca1_df=pca1.fit_transform(x_train_scaled)
            test_pca1_df=pca1.transform(x_test_scaled)
            

            ################################################################################################
            # Creating a combine pickle for both PCA and Standard Scaler to later use in the pipelinle

            data_columns = x_train.columns

            '''

            preprocessor = ColumnTransformer(
                transformers=[
                    ('converting the data into standard scaler', StandardScaler(), data_columns),  # StandardScaler for numeric columns
                    ('Reducing the Demension with pca', PCA(n_components=2), data_columns)  # PCA for all columns
                ],
                remainder='drop'  # Drop columns not included in transformers
            )
            '''

            with open('transformation.pkl', 'wb') as f:
                pickle.dump(pca1,f) 
            
            # saving it to the directory

            logging.info('Data Successfully transformed saving it now to the tranformed directory location')


            current_date_time = datetime.now().strftime("%d_%m_%H_%M_%S_")
            directory = os.path.join(dir , 'transformed_obj',current_date_time)
            os.makedirs(directory, exist_ok= True)
            final_location_of_transformed_obj = os.path.join(directory,'transformation.pkl')

            shutil.move('transformation.pkl', directory)

            train_df_scaled_final= pd.DataFrame(train_pca1_df, columns = [ 'scaled '+ str(i) for i in range(1,7)])

            test_df_scaled_final= pd.DataFrame(test_pca1_df, columns = [ 'scaled '+ str(i) for i in range(1,7)])

            # Saving the data sets to the artifact folder directory

            # creating directories to save these files


        # **********************************************************8


            artifact_time_stamp_loc =  self.artifact_time_stamp_loc
            transformed_directory_train = os.path.join(artifact_time_stamp_loc, 'Transformed_data', 'train_transformed_data')
            os.makedirs(transformed_directory_train, exist_ok=True)
            transformed_directory_test = os.path.join(artifact_time_stamp_loc, 'Transformed_data', 'test_transformed_data')
            os.makedirs(transformed_directory_test , exist_ok=True)


            saving_location_train = os.path.join(transformed_directory_train,"transf_X_train.csv")
            saving_location_test=os.path.join(transformed_directory_test,"transf_X_test.csv")

            # Saving the y_train and Y_test to the transformed dir 

            y_train_loc_transformed = os.path.join(transformed_directory_train , "transf_target_train.csv" )
            
            y_train.to_csv(y_train_loc_transformed)



            y_test_loc_transformed = os.path.join(transformed_directory_test , "transf_target_test.csv" )

            y_test.to_csv(y_test_loc_transformed)



            # saving this files to the above created locations


            saving_location_train = r"\\".join(saving_location_train.split("\\")) 

            saving_location_test = r"\\".join(saving_location_test.split("\\"))   
    


            train_df_scaled_final.to_csv(saving_location_train)
            test_df_scaled_final.to_csv(saving_location_test)


            # Creating Pandas profile report for the transformed data set 

            logging.info('Creating Profile report of our transformed data')


            # making artifact directories

            profile_report_location_for_html = "Templates\\PandasProfiling_after"

            
            destination_profile_report_dir_loc = os.path.join(self.artifact_time_stamp_loc  , "Transformed_Data_Analysis_report")
            os.makedirs(destination_profile_report_dir_loc, exist_ok = True)

            # Initiating profile report 

            profile_report_location = os.path.join(destination_profile_report_dir_loc, 'Transformed_Pandas_profiling.html')

            profile = ProfileReport(train_df_scaled_final)

            
            # Generate the report and save it
            profile.to_file(profile_report_location)

            #moving this file to the new location inside template for html access

            shutil.copy(profile_report_location,profile_report_location_for_html)

        except Exception as e:
            raise HousingException(e , sys) from e


        # Assigning values to the entity and returning as an output

        data_tranformation_entity_output = data_tranformation_entity(transformed_x_train_loc=saving_location_train  ,
        y_train_loc=y_train_loc_transformed, tranformed_x_test_loc=saving_location_test, 
        y_test_loc=y_test_loc_transformed, Z_score_loc = file_location_z_score, transformed_obj_loc = final_location_of_transformed_obj)

        logging.info(f'Data Transformation performed successfully artifact information are {data_tranformation_entity_output}')

        return data_tranformation_entity_output


 




 












'''
train_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\California Housing Price Prediction 2020\\\\artifact\\\\2023-09-09_16-09-08\\\\train_test_data\\\\train_data\\\\train_input.csv'
train_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\California Housing Price Prediction 2020\\\\artifact\\\\2023-09-09_16-09-08\\\\train_test_data\\\\train_data\\\\train_target.csv'
test_input_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\California Housing Price Prediction 2020\\\\artifact\\\\2023-09-09_16-09-08\\\\train_test_data\\\\test_data\\\\test_input.csv'
test_target_loc='C:\\\\Users\\\\Lokesh\\\\Desktop\\\\California Housing Price Prediction 2020\\\\artifact\\\\2023-09-09_16-09-08\\\\train_test_data\\\\test_data\\\\test_target.csv'
artifact_time_stamp_loc='C:\\Users\\Lokesh\\Desktop\\California Housing Price Prediction 2020\\artifact\\2023-09-09_16-09-08'
obj = data_tranformation(train_input_loc,train_target_loc,test_input_loc,test_target_loc,4,30,artifact_time_stamp_loc)
print(obj.initiate_data_transformation())
'''







