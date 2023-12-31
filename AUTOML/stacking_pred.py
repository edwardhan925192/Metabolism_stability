import pandas as pd
import math

def recursive_training_and_prediction(model_name, X, y, best_params, n_iterations=10):
    n = len(X)
    val_size = math.ceil(0.1 * n)
    predictions = []

    for i in range(0, n, val_size):
        start_val = i
        end_val = i + val_size

        if end_val >= n:  # Adjust end_val if we're at the end of the dataset
            end_val = n

        X_val = X.iloc[start_val:end_val]
        y_val = y.iloc[start_val:end_val]

        X_train = pd.concat([X.iloc[:start_val], X.iloc[end_val:]])
        y_train = pd.concat([y.iloc[:start_val], y.iloc[end_val:]])

        if model_name == 'xgboost':
            dtrain = DMatrix(X_train, label=y_train)
            model = xgb_train(best_params, dtrain)
            preds = model.predict(DMatrix(X_val))

        elif model_name == 'catboost':
            model = CatBoostRegressor(**best_params)
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_val)

        elif model_name == 'lightgbm':
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(best_params, dtrain)
            preds = model.predict(X_val)

        predictions.extend(preds)  # store the predictions

    return predictions
