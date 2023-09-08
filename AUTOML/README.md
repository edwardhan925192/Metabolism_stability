# Main  
python main.py \  
--train_path train_path \    
--test_path test_path \  
--feature \  
--maccs \  
--finger \  
--drop_column HLM \  
--stack_train_path train_path.joblib \  
--stack_test_path test_path.joblib \  
--mode main \  
--time_limit 3600 * 4  

# Colab  
!pip install autogluon  
!pip install rdkit  
!git clone https://github.com/edwardhan925192/Metabolism_stability.git  
%cd '/content/Metabolism_stability/AUTOML'  
Runtime should be restarted after installing autogluon   

* Data  
Data is expected to be csv files.  
* Features  
Feature,maccs,finger concatenate original dataframe with features of mols, maccs of mols,and morgan finger prints of mols.  
* Target  
If the Target is MLM drop HLM, if it is HLM drop MLM.    
* Stacking  
Stackings are expected to be in joblib form. if needed pass the paths.  
* Save mode  
If main is passed prediction is saved in csv form with id. If sub is passed prediction is saved in joblib format.  
* Timelimit  
Time limit of autogluon

# Stacking  



