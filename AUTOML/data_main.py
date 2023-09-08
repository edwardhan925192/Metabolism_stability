import argparse
import pandas as pd
from data_prep_main import preprocess_call_data
from morgan_finger_features import generate_morgan_fingerprints_and_concat
from maccs_features import generate_and_concatenate_MACCS_keys
from mol_features import data_prep

def main(args):
    train, test = preprocess_call_data(args.train_path, args.test_path)

    if args.feature:
        trainf = data_prep(train)
        testf = data_prep(test)
    else:
        trainf, testf = train, test

    drop_column = args.drop_column  # 'HLM' or 'MLM'

    if args.maccs:
        trainf = generate_and_concatenate_MACCS_keys(trainf, drop_columns=[drop_column])
        testf = generate_and_concatenate_MACCS_keys(testf)

    if args.finger:
        trainf = generate_morgan_fingerprints_and_concat(train)
        testf = generate_morgan_fingerprints_and_concat(test)

    trainf = trainf.drop(['id'], axis=1)
    testf = testf.drop(['id'], axis=1)

    # Save the results as CSV files
    trainf.to_csv('trainf.csv', index=False)
    testf.to_csv('testf.csv', index=False)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecular data with different feature sets.")

    parser.add_argument("--feature", action="store_true", help="Whether to use features or not.")
    parser.add_argument("--maccs", action="store_true", help="Whether to use MACCS keys.")
    parser.add_argument("--finger", action="store_true", help="Whether to use morgan fingerprints.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--drop_column", type=str, required=True, choices=['HLM', 'MLM'], help="Column to drop. Choose between 'HLM' and 'MLM'.")

    args = parser.parse_args()
    main(args)