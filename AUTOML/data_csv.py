import argparse
import pandas as pd
import joblib
from data_prep_main import preprocess_call_data
from morgan_finger_features import generate_morgan_fingerprints_and_concat
from maccs_features import generate_and_concatenate_MACCS_keys
from mol_features import data_prep
from similarity_features import compute_similarity, add_similarity_features
from Zagreb_index import calculate_zagreb_index

def main(args):
    train, test = preprocess_call_data(args.train_path, args.test_path)

    if args.feature:
        trainf = data_prep(train)
        testf = data_prep(test)
    else:
        trainf, testf = train, test    

    if args.maccs:
        trainf = generate_and_concatenate_MACCS_keys(trainf)
        testf = generate_and_concatenate_MACCS_keys(testf)

    if args.finger:
        trainf = generate_morgan_fingerprints_and_concat(train)
        testf = generate_morgan_fingerprints_and_concat(test)

    if args.similarity:
        trainf = add_similarity_features(trainf, args.nBits)
        testf = add_similarity_features(testf, args.nBits)

    if args.zagreb:
        trainf = calculate_zagreb_index(trainf)
        testf = calculate_zagreb_index(testf)

    trainf = trainf.drop(['id','SMILES'], axis=1)
    testf = testf.drop(['id','SMILES'], axis=1)

    if args.stack_train_path and args.stack_test_path:
        train_stacking_preds = joblib.load(args.stack_train_path)
        test_stacking_preds = joblib.load(args.stack_test_path)

        trainf['stacking_preds'] = train_stacking_preds
        testf['stacking_preds'] = test_stacking_preds
    
    # Save datasets with features to CSV
    trainf.to_csv('train_features.csv', index=False)
    testf.to_csv('test_features.csv', index=False)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecular data with different feature sets.")

    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--feature", action="store_true", help="Whether to use features or not.")
    parser.add_argument("--maccs", action="store_true", help="Whether to use MACCS keys.")
    parser.add_argument("--finger", action="store_true", help="Whether to use morgan fingerprints.")    
    parser.add_argument("--similarity", action="store_true", help="Whether to compute molecular similarity.")  
    parser.add_argument("--nBits", type=int, default=1024, help="Number of bits for Morgan fingerprint.")      
    parser.add_argument("--zagreb", action="store_true", help="Whether to concatenate zagreb index.")  
    parser.add_argument("--stack_train_path", type=str, help="Path to the joblib file containing stacking predictions for the train set.")
    parser.add_argument("--stack_test_path", type=str, help="Path to the joblib file containing stacking predictions for the test set.")    
    
    args = parser.parse_args()
    main(args)
