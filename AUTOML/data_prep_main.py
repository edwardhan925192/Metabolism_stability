import pandas as pd

def preprocess_call_data(train_path, test_path):
    """
    Loads and preprocesses the train and test data.
    
    Returns:
    - train: Preprocessed training dataframe.
    - test: Preprocessed testing dataframe.
    """
    
    # Paths (assuming these are constants for your project)
    train_data_path = train_path
    test_data_path = test_path
    
    # Load data
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    
    # Preprocess data
    train.loc[2796, 'AlogP'] = train.loc[2796, 'LogD']
    train.loc[3387, 'AlogP'] = train.loc[3387, 'LogD']
    test.loc[10, 'AlogP'] = test.loc[10, 'LogD']
    
    return train, test
