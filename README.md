# CTMEG: A Continuous-Time Medical Event Generation Model for Clinical Prediction of Long-Term Disease Progression

## 1 Introduction
This is an open-sourced project for CTMEG (a continuous-time medical event generation model for clinical prediction of  long-term disease progression
). 

## 2 Data preprocessing
We publish the source codes for preprocessing the [eICU](https://eicu-crd.mit.edu/gettingstarted/access/) database. 
1. Download the eICU database.
2. ```cd ./data```
3. Set the ```db_path``` and ```target_dir```, then run ```python main_dataset.py --db_name eicu --db_path [your path] --target_dir [your directory]```, the processed json data will be saved into the target directory.

## 3 Train
Set the following parameters in the script run.sh:
- train_path: the path of processed training json data.
- eval_path:  the path of processed validation json data.
- test_path:  the path of processed testing data.
- vocab_path: the path of the vocabulary of medical events including diagnosis, medication and laboratory tests.
- model_save_dir: the directory of pytorch model to be saved.

Then, run ```bash run.sh``` in the terminal.

## 4 Test
To obtain the final test results, simply comment out the training code in the run.sh script and re-execute it.
