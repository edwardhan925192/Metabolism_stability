import argparse
import pandas as pd
from data_prep_main import preprocess_call_data
from morgan_finger_features import generate_morgan_fingerprints_and_concat
from maccs_features import generate_and_concatenate_MACCS_keys
from mol_features import data_prep
from stacking_optim import optimize_hyperparams
from stacking_pred import recursive_training_and_prediction
from stacking_concat import predict_on_test

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

    # Drop the 'id' and 'SMILES' columns
    trainf = trainf.drop(['id', 'SMILES'], axis=1)
    testf = testf.drop(['id', 'SMILES'], axis=1)

    # Optimization and predictions
    if drop_column == "MLM":
        target_column = "HLM"
    else:
        target_column = "MLM"
    
    X = trainf.drop([target_column],axis = 1)
    y = trainf[target_column]
    
    best_param = optimize_hyperparams(args.model, X, y, args.optuna_trials)
    train_pred = recursive_training_and_prediction(args.model, X, y, best_param)
    test_pred = predict_on_test(args.model, X, y, testf, best_param)


    # Additional tasks can be added here if needed
    suffix = 'MLM' if drop_column == 'HLM' else 'HLM'

    # Save the predictions
    joblib.dump(train_pred, f'{args.model}_{suffix}.joblib')
    joblib.dump(test_pred, f'{args.model}_{suffix}.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecular data with different feature sets.")

    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--feature", action="store_true", help="Whether to use features or not.")
    parser.add_argument("--maccs", action="store_true", help="Whether to use MACCS keys.")
    parser.add_argument("--finger", action="store_true", help="Whether to use morgan fingerprints.")
    parser.add_argument("--drop_column", type=str, required=True, choices=['HLM', 'MLM'], help="Column to drop. Choose between 'HLM' and 'MLM'.")
    parser.add_argument("--model", type=str, required=True, choices=['lightgbm', 'xgboost', 'catboost'], help="Model to use for predictions.")
    parser.add_argument("--optuna_trials", type=int, default=50, help="Number of trials for Optuna optimization.")

    args = parser.parse_args()
    main(args)
