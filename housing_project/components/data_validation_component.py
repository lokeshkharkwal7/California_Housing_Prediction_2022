# Reading the training and testing data from the ingestion step
# Reading the contents of the schema files
# matching the column names of our dataset with the column names that is provided in our Schema file
# Converting the datatypes of our dataset with our schema file
# checking the datadrift using evidently
# validating and sending it to the other step


from housing_project.components import data_ingestion_component
from housing_project.entity.components_entity import data_validation_entity
import os,sys
from utility.util import *
import pandas as pd
import shutil
from collections import Counter # used to compare the list
# imports for data drift
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pandas_profiling import ProfileReport
from housing_project.exception import HousingException
from housing_project.logger import logging

from housing_project.entity.components_entity import data_validation_entity




class data_validation:
    def __init__(self, X_train_location , Y_train_location , X_test_location , Y_test_location,
    artifact_time_stamp_loc,data_validation_entity):

         
         self.x_train_loc = X_train_location
         self.y_train_loc = Y_train_location
         self.x_test_loc = X_test_location
         self.y_test_loc = Y_test_location
         self.artifact_time_stamp_loc =artifact_time_stamp_loc
         self.data_validation_entity = data_validation_entity

    def initiate_data_validation(self):

        try:
            # initiating flag variable 
            logging.info('Data Validation started')
            Column_matching_status = False
            Validation_status = False
            

            # Reading the training and testing file data frames with the parameters provided

            X_train = pd.read_csv(self.x_train_loc)
            Y_train = pd.read_csv(self.y_train_loc)

            X_test = pd.read_csv(self.x_test_loc)
            Y_test = pd.read_csv(self.y_test_loc)

            # Removing Unnamed: 0 from train and test if problem perisist while reading the databases

            if 'Unnamed: 0' in X_train.columns:
                X_train.drop(columns='Unnamed: 0', inplace = True)
            if 'Unnamed: 0' in Y_train.columns:
                Y_train.drop(columns = 'Unnamed: 0', inplace = True)

            if 'Unnamed: 0' in X_test.columns:
                X_test.drop(columns='Unnamed: 0', inplace = True)
            if 'Unnamed: 0' in Y_test.columns:
                Y_test.drop(columns = 'Unnamed: 0', inplace = True)
            



            
            # Reading the contents of the schema file 
            logging.info('Comparing Schema of the downloaded data set')
        
            config_data = read_config('config.yaml')
            schema_file_loc = config_data['Data validation']['schema']
            schema_file= read_config(schema_file_loc)

            schema_input_columns = schema_file['numerical_columns']
            schema_output_columns = schema_file['target_columns']

        except Exception as e:
            raise HousingException(e, sys) from e

         # Comparing the column name of our schema_files with the coulumns of our X_train and Y_train

         # Used Counter because it will help us to compare the different list
         
        def column_compare(train_df , test_df) :  

    
            if Counter(train_df.columns) == Counter(schema_input_columns):
                Validation_status = True
                print("Schema of input features matched")
            else:
                Validation_status = False
                print(" Columns names are incorrect ")

            if Validation_status == True:

                if  test_df.columns == schema_output_columns:
                    Validation_status = True
                    print('Schema of output features are matched')
                    return Validation_status
                else:
                    Validation_status = False

                    print('Schema of output features does not matched')

                    return Validation_status
        try:

            flag_train = column_compare(X_train,Y_train)
            flag_test = column_compare(X_test, Y_test)
        except Exception as e:
            raise HousingException(e, sys) from e

        # Changing data types as per the schema.yaml file 

        logging.info(f'Comparing Status: completed')
        logging.info('Changing Data Types of the downloaded data set')
        
        def change_dtypes(train_df , test_df):

            # Reading the data types of the columns from schema.yaml file

            dtypes_in_schema = schema_file['columns']
                
                    
            # Changing the dtypes of our features as per the schema file

            print('data types before: ')
            print(train_df.dtypes)

            # dtypes_in_schema [i] will tell the data types that are mentioned in the schema.yaml file 
            # for each and every columns and i is the name of the columns present in the data set

            for i in train_df.columns:

                train_df[i] = train_df[i].astype(dtypes_in_schema[i])

            for i in test_df.columns:
                test_df[i] = test_df[i].astype(float)


                print('\n')

                print("data type after is: ")
                print(train_df.dtypes)


        if (flag_train and flag_test):


                try:
                    change_dtypes(X_train, Y_train)
                    change_dtypes(X_test, Y_test)
                except Exception as e:
                    raise HousingException(e, sys) from e

          



        # Checking the data drift for training and testing data sets

        def data_drift_status()->bool: 

            
                logging.info('Successfully changed the Dtypes Checking data drift')
                report = Report(metrics=[DataDriftPreset(), ])

                report.run(reference_data=X_train, current_data=X_test)
                report.save_html('Data_Drift_Report.html')
                
                metrics_list = report.as_dict()['metrics']
                df_rows = []

                for metric_data in metrics_list:
                    metric = metric_data['metric']
                    result = metric_data['result']
                    df_rows.append({'metric': metric, **result})

                df = pd.DataFrame(df_rows)
                for i in df[['dataset_drift']].loc[1]:
                    answer = i
                return answer

        try:
            if data_drift_status()==True:
                print('Data Drift Detected')
                logging.info('Data Drift Detected')
                Validation_status = False
            else:
                print('No Data Drift Detected good to go')
                logging.info('No Data Drift Detected good to go')
                Validation_status=True

        except Exception as e:
            raise HousingException(e, sys) from e

        # run evidently and generate the output report 

        try:

            source_path = os.path.abspath('Data_Drift_Report.html')

            destination_evidently_report_loc = os.path.join(self.artifact_time_stamp_loc , "Data_Drift_report")
            os.makedirs(destination_evidently_report_loc, exist_ok = True)

        except Exception as e:
            raise HousingException(e, sys) from e
         
         # moving evidently report file to the right location where our data ingestion 
         # files are kept
    
        try:
                shutil.move(source_path,destination_evidently_report_loc)

        except Exception as e:
            print(e)

        try:

            if Validation_status == True:

                print('You can check the report on this location: ', destination_evidently_report_loc)
                logging.info('You can check the report on this location: ', destination_evidently_report_loc)


            # Using Pandas Profiling to get the data analysis report 


            # making artifact directories

            profile_report_location_for_html = os.path.join(root_dir,'Templates','PandasProfiling_before')

            
            destination_profile_report_dir_loc = os.path.join(self.artifact_time_stamp_loc , "Data_Analysis_report")
            os.makedirs(destination_profile_report_dir_loc, exist_ok = True)

            # Initiating profile report 

            profile_report_location = os.path.join(destination_profile_report_dir_loc, 'Pandas_profiling.html')

            profile = ProfileReport(X_train)

            
            # Generate the report and save it
            profile.to_file(profile_report_location)

            #moving this file to the new location inside template for html access

            shutil.copy(profile_report_location,profile_report_location_for_html)






            # Storing output in Entity which is the output of the Data_validation 

            data_validation_entity = self.data_validation_entity(is_validated=Validation_status, Data_drift_file_location=destination_evidently_report_loc)

            logging.info('Data Validatin Status : ', data_validation_entity)
            return data_validation_entity
        
        except Exception as e:
            raise HousingException(e, sys) from e

'''
         
X_train_location= "artifact\\01_09_14_54_23_\\train_test_data\\train_data\\train_input.csv"
Y_train_location= "artifact\\01_09_14_54_23_\\train_test_data\\train_data\\train_target.csv"
X_test_location= "artifact\\01_09_14_54_23_\\train_test_data\\test_data\\test_input.csv"
Y_test_location= "artifact\\01_09_14_54_23_\\train_test_data\\test_data\\test_target.csv"
artifact_time_stamp_loc='artifact\\01_09_14_54_23_'
data_validation_entity= data_validation_entity
obj = data_validation(X_train_location , Y_train_location , X_test_location , Y_test_location,
    artifact_time_stamp_loc,data_validation_entity)

print(obj.initiate_data_validation())

'''








        



        
          

        
