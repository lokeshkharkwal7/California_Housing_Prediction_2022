# creating a fuction that will take the input value as a data frame and use our pickle files
# transformation and modelling to print the result

from utility.util import * 
import pickle

class output_generation :
    def __init__(self , data):
        self.data = data 

    def generate_output(self):
    # reading all pickle files 

      
        with open( os.path.join(root_dir, 'Pipeline Models and Transformers','z_score_scaling.pkl') , 'rb') as f:
             z_scaling = pickle.load(f)

        with open(os.path.join(root_dir, 'Pipeline Models and Transformers','pca.pkl')  , 'rb') as f:
              pca = pickle.load(f)

        with open(os.path.join(root_dir, 'Pipeline Models and Transformers\Best_model.pkl')  , 'rb') as f:
            model = pickle.load(f)
      


       
    

    # converting the data into z score and pca:
        z_scaled_df = z_scaling.transform(self.data)
        pca_df = pca.transform(z_scaled_df)

    # predicting the output from the model
        output = float(model.predict(pca_df))
    

        return (output)


'''
        transformed_obj_loc='C:\\Users\\Lokesh\\Desktop\\California Housing Price Prediction 2020\\transformed_obj\\09_09_16_10_46_\\transformation.pkl'
        Z_score_loc='C:\\Users\\Lokesh\\Desktop\\California Housing Price Prediction 2020\\Z_Score_Obj\\Z_score.pkl'

        with open( transformed_obj_loc  , 'rb') as f:
            pca = pickle.load(f)

        with open(Z_score_loc, 'rb') as f:
            z_scaling = pickle.load(f)
'''

 

        

    





     
