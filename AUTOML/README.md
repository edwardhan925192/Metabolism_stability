# mol_features.py
from mol_features import data_prep  
data_prep(train)  

dataframe is concatted with features of mols  

# maccs_features.py  
from maccs_features import generate_and_concatenate_MACCS_keys  
generate_and_concatenate_MACCS_keys(trainf, drop_columns=['HLM'])  
generate_and_concatenate_MACCS_keys(testf)  

dataframe is concatted with maccs of mols  

# morgan_finger_features.py  
from morgan_finger_features import generate_morgan_fingerprints_and_concat  
concatenated_train = generate_morgan_fingerprints_and_concat(train)  
concatenated_test = generate_morgan_fingerprints_and_concat(test)  

dataframe is concatted with morgan_finger_prints of mols    

# data_prep_main.py   
from data_prep_main import preprocess_call_data  
train, test = preprocess_data()  

Missing values are filled and dataframe is called  
