import argparse
import pandas as pd
import joblib
from data_prep_main import preprocess_call_data
from morgan_finger_features import generate_morgan_fingerprints_and_concat
from maccs_features import generate_and_concatenate_MACCS_keys
from mol_features import data_prep
from autogluon.tabular import TabularPredictor

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

    trainf = trainf.drop(['id','SMILES'], axis=1)
    testf = testf.drop(['id','SMILES'], axis=1)

    if args.stack_train_path and args.stack_test_path:
        train_stacking_preds = joblib.load(args.stack_train_path)
        test_stacking_preds = joblib.load(args.stack_test_path)

        trainf['stacking_preds'] = train_stacking_preds
        testf['stacking_preds'] = test_stacking_preds
    
    # Training and Prediction with AutoGluon
    label = 'HLM' if args.drop_column == 'MLM' else 'MLM'
    predictor = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
        trainf,
        presets='best_quality',
        time_limit=args.time_limit
    )
    predictions = predictor.predict(testf)

    # Saving Predictions
    if args.mode == 'main':
        predictions.to_csv('predictions.csv', index=False)
    elif args.mode == 'sub':
        joblib.dump(predictions, 'predictions.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecular data with different feature sets.")
    
    parser.add_argument("--feature", action="store_true", help="Whether to use features or not.")
    parser.add_argument("--maccs", action="store_true", help="Whether to use MACCS keys.")
    parser.add_argument("--finger", action="store_true", help="Whether to use morgan fingerprints.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--drop_column", type=str, required=True, choices=['HLM', 'MLM'], help="Column to drop. Choose between 'HLM' and 'MLM'.")
    parser.add_argument("--stack_train_path", type=str, help="Path to the joblib file containing stacking predictions for the train set.")
    parser.add_argument("--stack_test_path", type=str, help="Path to the joblib file containing stacking predictions for the test set.")
    parser.add_argument("--mode", type=str, required=True, choices=['main', 'sub'], 
                        help="Mode of operation. 'main' will save predictions in CSV. 'sub' will save predictions in joblib format.")
    parser.add_argument("--time_limit", type=int, default=3600 * 4.5, help="Time limit for AutoGluon in seconds.")
    
    args = parser.parse_args()
    main(args)
