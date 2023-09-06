# mol_features.py
from mol_features import data_prep  
data_prep(train)  

dataframe is concatted with features of mols

# maccs_features.py  
from maccs_features import generate_and_concatenate_MACCS_keys  
generate_and_concatenate_MACCS_keys(trainf, drop_columns=['HLM'])  
generate_and_concatenate_MACCS_keys(testf)  

dataframe is concatted with maccs of mols

# data_prep_main.py   
import preprocess_data  
train, test = preprocess_data()  

Missing values are filled dataframe is called 
