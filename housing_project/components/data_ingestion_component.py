 #creating artifact directory
    # Creating a new directory based on the time stamp
     # creating a new directory to save extracted data
     # Creating a file path of the name of the file you are extracting
       # Downloading the file to the location of the name of the file you have created on the above step.
       # creating a new folder to save the data of the tar file ( The file you have downloaded in the above step)
       # extracting this file to the given file location ( location name : csv_file_location )
       # Using pandas to read and split the data into train and test 
       # creating a seprate folder to store training and testing files
       # Moving the training and testing data to the following location ( you also need to create sperate training and target files inside)
       # First converted the file location url to raw string and then moved to specific location since pandas does not accept any other url
       # Save the location of train x , train y , test x , test y to our named tuple ENTITY of data_ingestion


import os,sys
import urllib.request
from datetime import datetime
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
from housing_project.entity.components_entity import *
from housing_project.configuration.housing_configuration import ingestion_configuration
from housing_project.exception import HousingException
import shutil
from utility.util import *

from housing_project.logger import logging


 
ingestion_config = ingestion_configuration()
download_url = ingestion_config.url_key()
class data_ingestion_component:


    def __init__(self , url=download_url):
        self.url = url
    
    def initiate_data_ingestion(self):


        try:

        # Deleting the logging folder : 



        # Creating artifact dir 

            artifact_dir_loc = os.path.join(root_dir, 'artifact')
            os.makedirs(artifact_dir_loc , exist_ok=True)

            # creating a new dirctory based on time stamp

            time_stamp_dir_loc = os.path.join(artifact_dir_loc , datetime.now().strftime('20%y-%d-%m_%H-%M-%S'))
            os.makedirs(time_stamp_dir_loc , exist_ok=True)


            # creating a new dirctory to save the extracted data ( file name : raw_data)

            raw_data_dir_loc = os.path.join(time_stamp_dir_loc, 'raw_data')
            os.makedirs(raw_data_dir_loc, exist_ok = True)

            # creating a new file location which also contains the name of your file 

            download_file_location = os.path.join(raw_data_dir_loc, 'housing.tgz')

            logging.info('Data Ingestion Directory successfully created')

        except Exception as e:
            raise HousingException(e, sys) from e

        # downloading a file to the above locaton created ( on folder : raw_data )

        try:
            logging.info('Downloading the files')

            download_url = self.url
            urllib.request.urlretrieve(download_url, download_file_location )

            logging.info('file downloaded successfully')


        # creating a new folder to save the data of the tar file 

            csv_file_location = os.path.join(time_stamp_dir_loc , 'csv_file')
            os.makedirs(csv_file_location , exist_ok=True)

        except Exception as e:
            raise HousingException(e , sys) from e
        

        # extracting this file to the given file location ( csv_file_location )

        try: 

            logging.info('Extracting the Zip file')


            with tarfile.open( download_file_location , 'r') as tar:
                tar.extractall(csv_file_location)

            logging.info('Zip file extracted')

        except Exception as e:
            raise HousingException(e, sys) from e

        # opening the csv and extracting it into train and test_split

 
        file_location = os.path.join(csv_file_location, "CaliforniaHousing\\cal_housing.data")

        # using r join since pandas only takes raw string 

        raw_dynamic_path = r"\\".join(file_location.split("\\"))

        # assigning column names to our dataset

        try:

            logging.info('Assigning column names to the CSV file')

            data = pd.read_csv(raw_dynamic_path)
            data.columns = ['longitude', 'latitude' , 'housingMedianAge', 'totalRooms', 'totalBedrooms', 'population','households','medianIncome','medianHouseValue']
    
    
            input = data.drop(columns = 'medianHouseValue')
            target = data['medianHouseValue']
            target = pd.DataFrame(target)

        except Exception as e:
            raise HousingException(e, sys) from e
 


        # spliting the data YOU CAN ALSO USE STRATIFY ATTRIBUTE FOR TRAIN_TEST_SPLIT IF YOU ARE HANDELING
        # CATAGORICAL COLUMNS

        try:

            X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.10, random_state=42)

            # creating a seprate folder to store training and testing files

            logging.info('Seperating Training and Testing Data')

            train_test_location = os.path.join(time_stamp_dir_loc , 'train_test_data')
            train_data_location = os.path.join(train_test_location, "train_data")
            os.makedirs(train_data_location)

            test_data_location = os.path.join(train_test_location, 'test_data')
            os.makedirs(test_data_location)

            # Moving the training and testing data to the following location name of file is
            #imp else it will show error.

            X_location_for_train_input = os.path.join(train_data_location,'train_input.csv')
            Y_location_for_train_input = os.path.join(train_data_location, 'train_target.csv')

            X_test_input_location = os.path.join(test_data_location,"test_input.csv")
            Y_test_input_location = os.path.join(test_data_location, "test_target.csv")


            # using r join since pandas only takes raw string on train test

            X_location_for_train_input = r"\\".join(X_location_for_train_input.split("\\"))   

            Y_location_for_train_input = r"\\".join(Y_location_for_train_input.split("\\"))



            # using r join since pandas only takes raw string on test test

            X_test_input_location = r"\\".join(X_test_input_location.split("\\"))
            Y_test_input_location = r"\\".join(Y_test_input_location.split("\\"))

        except Exception as e:
            raise HousingException(e , sys) from e



        try:
            X_train.to_csv( X_location_for_train_input)
            y_train.to_csv(Y_location_for_train_input)

            X_test.to_csv( X_test_input_location)
            y_test.to_csv(Y_test_input_location)
            logging.info('Training and testing data seperated successfully')

        except Exception as e:
            raise HousingException(e, sys) from e

        data_ingestion_output = data_ingestion_entity(train_input_loc=X_location_for_train_input,
        train_target_loc=Y_location_for_train_input , test_input_loc=X_test_input_location,
        test_target_loc=Y_test_input_location, artifact_time_stamp_loc = time_stamp_dir_loc)

        return data_ingestion_output

'''
obj = data_ingestion_component('https://figshare.com/ndownloader/files/5976036')
print(obj.initiate_data_ingestion())
'''
 
    


