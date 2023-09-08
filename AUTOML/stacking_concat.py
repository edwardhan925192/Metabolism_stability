from stacking_optim import optimize_hyperparams
from stacking_pred import recursive_training_and_prediction

def predict_on_test(model_name, X_train, y_train, X_test, best_params):

    if model_name == 'xgboost':
        dtrain = DMatrix(X_train, label=y_train)
        model = xgb_train(best_params, dtrain)
        predictions = model.predict(DMatrix(X_test))

    elif model_name == 'catboost':
        model = CatBoostRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)
        predictions = model.predict(X_test)

    elif model_name == 'lightgbm':
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(best_params, dtrain)
        predictions = model.predict(X_test)

    return predictions

# defining X and y
X = trainf.iloc[:,2:]
y = trainf['MLM']

# optimization
best_param_lgb = optimize_hyperparams('lightgbm',X,y,50)

# prediction for train
lgb_pred = recursive_training_and_prediction('lightgbm',X,y,best_param_lgb)

# prediction for test
lgb_pred_test = predict_on_test(X,y,testf,best_param_lgb)

# concat train test
trainf_stack = pd.concat([trainf,lgb_pred], axis = 1)
testf_stack = pd.concat([testf,lgb_pred_test], axis = 1)
