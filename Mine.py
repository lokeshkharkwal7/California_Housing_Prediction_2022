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
       # First converted the file location url to raw string and then moved to specific location since pandas does not accept any other ur






import os
import urllib.request
from datetime import datetime
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split




# Creating artifact dir 

artifact_dir_loc = os.path.join(os.getcwd() , 'artifact')
os.makedirs(artifact_dir_loc , exist_ok=True)

# creating a new dirctory based on time stamp

time_stamp_dir_loc = os.path.join(artifact_dir_loc , datetime.now().strftime("%d_%m_%H_%M_%S_"))
os.makedirs(time_stamp_dir_loc , exist_ok=True)


# creating a new dirctory to save the extracted data ( file name : raw_data)

raw_data_dir_loc = os.path.join(time_stamp_dir_loc, 'raw_data')
os.makedirs(raw_data_dir_loc, exist_ok = True)

# creating a new file location which also contains the name of your file 

download_file_location = os.path.join(raw_data_dir_loc, 'housing.tgz')

# downloading a file to the above locaton created ( on folder : raw_data )

download_url = "https://figshare.com/ndownloader/files/5976036"
urllib.request.urlretrieve(download_url, download_file_location )

# creating a new folder to save the data of the tar file 

csv_file_location = os.path.join(time_stamp_dir_loc , 'csv_file')
os.makedirs(csv_file_location , exist_ok=True)
 

# extracting this file to the given file location ( csv_file_location )
with tarfile.open( download_file_location , 'r') as tar:
    tar.extractall(csv_file_location)

# opening the csv and extracting it into train and test_split

file_location = os.path.join(csv_file_location, "CaliforniaHousing\\cal_housing.data")

# using r join since pandas only takes raw string 

raw_dynamic_path = r"\\".join(file_location.split("\\"))


data = pd.read_csv(raw_dynamic_path)
data.columns = ['longitude', 'latitude' , 'housingMedianAge', 'totalRooms', 'totalBedrooms', 'population','households','medianIncome','medianHouseValue']

input = data.drop(columns = 'medianHouseValue')
target = data['medianHouseValue']

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.33, random_state=42)

# creating a seprate folder to store training and testing files

train_test_location = os.path.join(time_stamp_dir_loc , 'train_test_data')
train_data_location = os.path.join(train_test_location, "train_data")
os.makedirs(train_data_location)

test_data_location = os.path.join(train_test_location, 'test_data')
os.makedirs(test_data_location)

# Moving the training and testing data to the following location

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




X_train.to_csv( X_location_for_train_input)
y_train.to_csv(Y_location_for_train_input)

X_test.to_csv( X_test_input_location)
y_test.to_csv(Y_test_input_location)








