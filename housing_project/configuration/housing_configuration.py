# Will only be used in the main pipeline to provide configuration to run the specific
#component of our pipeline

from housing_project.keys.key import *

class ingestion_configuration:
    def __init__(self):
        pass
    def url_key(self):
        return URLKEY

class validation_configuration:
    def __init__(self):
        pass
    def schema_file_path(self):
        return SCHEMA_PATH

class data_transformation_configuration:
    def __init__(self):
        pass
    def transformed_directory_name(self):
        return TRANSFORMED_DIR

class model_selection_configuration:
    def __init__(self):
        pass
    def get_underfitting_score(self):
        return UNDERFITTING_SCORE
    def get_model_container_dir(self):
        return MODEL_CONTAINER_DIR

class model_pusher_configuration:
    def __init__(self):
        pass
    def get_best_modeldir(self):
        return BEST_MODEL_DIR

 