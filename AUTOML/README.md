# Dataprep  
This concat original dataframe with features, (functional groups, total bonds, and etc), maccs, and morgan finger prints. Depending on drop column, HLM or MLM is deleted from the dataframe and 4 csv files are returned. 2 csv files are original csv files, and other 2 csv files are concatenated csv files with deletion of HLM or MLM.  

python data_main.py --feature --maccs --train_path /path/to/train.csv --test_path /path/to/test.csv --drop_column HLM  

