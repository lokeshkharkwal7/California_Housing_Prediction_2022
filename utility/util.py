import yaml
import os
import glob


# Reading data from the config file

def read_config(file_path) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Moving data to the config file

def write_config(data , file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data , file, default_flow_style=False)

root_dir = os.path.join(os.getcwd())
housing_project_dir = os.path.join(root_dir,'housing_project')


def model_entry_to_yaml(data):
    # how to edit data present in the config.yaml file ( first read and then update then push)
 # Creating an dict to update the config.yaml file
 
    # after reading the previous config.yaml file 

    file_path = 'model_status.yaml'
    yaml_data = read_config(file_path)


    # using the update function update the dict we get from the read_config

    yaml_data.update(data)   ###############3

    # pushing back the data to the config.yaml file
    write_config(yaml_data, file_path)


def get_best_model_from_yaml():

        # updating the information that is present in the config.yaml file

    
    # reading the config.yaml file
    file_path = 'model_status.yaml'
    yaml_data = read_config(file_path)


    # updating the data with the information of the candidates with the highest score

    # highest scoring info:

            # You can find anything in the dictionary using the below code

    highest_score = max([ i['Model_score'] for i in yaml_data.values()])
    highest_score_model = [ i for i in yaml_data.values()  if i.get('Model_score')==highest_score]

    # getting information of the highest model 

    yaml_data['Best Model']['Model_name'] = highest_score_model[0]['Model_name']

    yaml_data['Best Model']['Model_score'] = highest_score_model[0]['Model_score']

    yaml_data['Best Model']['Pickle_Location'] = highest_score_model[0]['Pickle_Location']

    yaml_data['Best Model']['best_model_parameters'] = highest_score_model[0]['best_model_parameters']


    # pushing this list back to the config.yaml file

    write_config(yaml_data , file_path)
    return highest_score_model[0]

def get_latest_file(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        return None

    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory, '*'))

    # Filter out directories and get the latest file based on its modification time
    latest_file = max(files, key=os.path.getctime, default=None)

    return latest_file

