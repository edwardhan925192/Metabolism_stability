import argparse
import pandas as pd
from autogluon.tabular import TabularPredictor
import joblib 

def main(args):
    # Load Data
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    test_ids = pd.read_csv(args.test_id_path)

    # Drop the specified column
    train = train.drop(args.drop_column, axis=1)
    test = test.drop(args.drop_column, axis=1)

    # Training and Prediction with AutoGluon
    label = 'HLM' if args.drop_column == 'MLM' else 'MLM'
    predictor = TabularPredictor(label=label, eval_metric='root_mean_squared_error').fit(
        train,
        presets='best_quality',
        time_limit=args.time_limit
    )
    predictions = predictor.predict(test)

    # Saving Predictions
    if args.mode == 'main':
        result_df = pd.DataFrame({
            'id': test_ids['id'], 
            'predictions': predictions
        })
        result_df.to_csv('predictions.csv', index=False)

    elif args.mode == 'sub':
        joblib.dump(predictions, 'predictions.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecular data.")

    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset without ids.")
    parser.add_argument("--test_id_path", type=str, required=True, help="Path to the CSV file containing the 'id' column for the test dataset.")
    parser.add_argument("--drop_column", type=str, required=True, choices=['HLM', 'MLM'], help="Column to drop. Choose between 'HLM' and 'MLM'.")
    parser.add_argument("--mode", type=str, required=True, choices=['main', 'sub'], 
                        help="Mode of operation. 'main' will save predictions in CSV. 'sub' will save predictions in joblib format.")
    parser.add_argument("--time_limit", type=int, default=3600 * 4.5, help="Time limit for AutoGluon in seconds.")
    
    args = parser.parse_args()
    main(args)
