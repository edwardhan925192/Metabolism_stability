import pandas as pd

def cyp_concat(train, test, train_path, test_path):    
    swiss1 = pd.read_csv(train_path)
    swiss2 = pd.read_csv(test_path)
    concat_df1 = pd.concat([train, swiss1],axis =1)
    concat_df2 = pd.concat([test, swiss2],axis =1)
    return concat_df1, concat_df2
