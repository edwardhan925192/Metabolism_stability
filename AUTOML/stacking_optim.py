import optuna
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import DMatrix, cv as xgb_cv, train as xgb_train
from catboost import CatBoostRegressor, Pool, cv as cat_cv
import lightgbm as lgb

def optimize_hyperparams(model_name, X, y, trials=100):

  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  if model_name == 'xgboost':
    def xgb_objective(trial):
      param = {
              'booster': 'gbtree',
              'objective': 'reg:squarederror',
              'n_estimators': trial.suggest_int('n_estimators', 10, 6000),
              'eta': trial.suggest_loguniform('eta', 0.005, 0.05),
              'max_depth': trial.suggest_int('max_depth', 1, 18),
              'subsample': trial.suggest_float('subsample', 0.9, 1.0),
              'min_child_weight': trial.suggest_float('min_child_weight', 0.0, 10.0),
              'lambda': trial.suggest_float('lambda', 0.01, 0.3),
              }
      dtrain = DMatrix(X, label=y)
      result = xgb_cv(param, dtrain, nfold=5, metrics="mae", as_pandas=True, seed=42)
      return result['test-mae-mean'].iloc[-1]

    study = optuna.create_study(direction='minimize')
    study.optimize(xgb_objective, n_trials=trials)
    return study.best_params

  elif model_name == 'catboost':
    def cat_objective(trial):
      param = {
                'loss_function': 'MSE',
                'iterations': trial.suggest_int('iterations', 10, 6000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
                'depth': trial.suggest_int('depth', 1, 18),
                'subsample': trial.suggest_float('subsample', 0.9, 1.0),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 0.3),
                'verbose': False
              }

      train_data = Pool(X, y)
      scores = cat_cv(pool=train_data, params=param, fold_count=5, seed=42, plot=False, as_pandas=True)
      return scores['test-MAE-mean'].iloc[-1]

    study = optuna.create_study(direction='minimize')
    study.optimize(cat_objective, n_trials=trials)
    return study.best_params

  elif model_name == 'lightgbm':
    def lgb_objective(trial):
      print("Entered lgb_objective...")
      param = {
              'boosting_type': 'gbdt',
              'objective': 'mse',  # Use custom objective function,
              'metric': 'l1',
              'n_estimators':trial.suggest_int('n_estimators', 10, 6000),
              'verbosity': -1,
              'extra_trees': True,
              'num_leaves': trial.suggest_int('num_leaves', 2, 100),
              'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
              'max_depth': trial.suggest_int('max_depth', 1, 18),
              'bagging_fraction': trial.suggest_float('bagging_fraction', 0.9, 1.0),  # Added upper limit,
              'path_smooth' : trial.suggest_float('path_smooth', 1, 25),
              'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 0.0, 10.0),
              'bagging_freq': 5,
              'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.3),
              'min_data_in_bin':1,
              'min_data_in_leaf':1
            }
      dtrain = lgb.Dataset(X, label=y)
      kf = KFold(n_splits=5, shuffle=True, random_state=42)
      folds = list(kf.split(X, y))
      print("Running lgb.cv()...")

      result = lgb.cv(param, dtrain, folds=folds, shuffle=True, seed=42)
      print("lgb.cv() ran successfully.")

      for key, value in result.items():
        print('*****'*10)
        print(f"{key}: {value[:5]}")

      return np.mean(result["valid l1-mean"])

    study = optuna.create_study(direction='minimize')
    study.optimize(lgb_objective, n_trials=trials)
    return study.best_params
