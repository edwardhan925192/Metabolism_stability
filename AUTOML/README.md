# Main  
python main.py \  
--train_path train_path \    
--test_path test_path \  
--test_id_path \  
--target HLM \  
--mode main \  
--time_limit 3600 * 4  

* Data  
Data is expected to be csv files.

* Test_id  
This id path is expected to be a csv files that contains id of the test.  

* Target  
There are 4 targets in total HLM, MLM, Diff(HLM - MLM), Mean

* Save mode  
If main is passed, prediction is saved in csv form with id. If sub is passed prediction is saved in joblib format.

* Timelimit  
Time limit of autogluon

# Colab  
!pip install autogluon  
!pip install rdkit  
!git clone https://github.com/edwardhan925192/Metabolism_stability.git  
%cd '/content/Metabolism_stability/AUTOML'  
Runtime should be restarted after installing autogluon   

# Stacking  
python stacking_main.py \  
--train_path path\to\train_data.csv \  
--test_path path\to\test_data.csv \  
--feature \  
--maccs \  
--drop_column HLM \  
--model lightgbm \  
--optuna_trials 50  

* Data  
Data is expected to be csv files.

* Features  
Feature,maccs,finger concatenate original dataframe with features of mols, maccs of mols,and morgan finger prints of mols.

* Target  
If the Target is MLM drop HLM, if it is HLM drop MLM.  

* Model  
Choose model for stacking  

* Trial  
Optuna trial

# Colab  
!pip install xgboost  
!pip install catboost 
!pip install lightgbm  
!pip install rdkit  
!git clone https://github.com/edwardhan925192/Metabolism_stability.git  
%cd '/content/Metabolism_stability/AUTOML'  
train_path  '/content/Metabolism_stability/AUTOML/train_ms.csv'  
test_path  '/content/Metabolism_stability/AUTOML/test_ms.csv'  

# CSV data prep  
python data_csv.py \  
--train_path '/content/Metabolism_stability/AUTOML/original_train.csv' \  
--test_path '/content/Metabolism_stability/AUTOML/original_test.csv' \  
--feature \  
--maccs \  
--similarity \  
--nBits 2048 \  
--zagreb \  
--train_path_cyp '/content/Metabolism_stability/AUTOML/data/swissaddme/swissaddme_train.csv' \  
--test_path_cyp '/content/Metabolism_stability/AUTOML/data/swissaddme/swissaddme_test.csv'  

# Colab  
!pip install rdkit  
!git clone https://github.com/edwardhan925192/Metabolism_stability.git  
%cd '/content/Metabolism_stability/AUTOML'  
Runtime should be restarted after installing autogluon   
train_path  '/content/Metabolism_stability/AUTOML/train_ms.csv'    
test_path  '/content/Metabolism_stability/AUTOML/test_ms.csv' 





