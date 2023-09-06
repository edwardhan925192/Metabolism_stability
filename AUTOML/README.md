# mol_features.py
import data_prep  
usage data_prep(train)  
dataframe is concatted with features of mols

# maccs_features.py  
import generate_and_concatenate_MACCS_keys  
Usage example  
generate_and_concatenate_MACCS_keys(trainf, drop_columns=['HLM'])  
generate_and_concatenate_MACCS_keys(testf)

# data_prep_main.py   
import preprocess_data  
Usage example  
train, test = preprocess_data()  
