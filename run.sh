seed=2024
batch_size=32
lr=2e-3
db_name=eicu
log_dir=./logger
train_path=xxx
eval_path=xxx
test_path=xxx
vocab_path=xxx
device=0
model_save_dir=xxx
model_name=multi_tpp_tl_lab
model_load_path=$model_save_dir$model_name.pth
epochs=30

# Train
CUDA_VISIBLE_DEVICES=$device python main.py --db_name $db_name --seed $seed --epochs $epochs --batch_size $batch_size --lr $lr --train_path $train_path --eval_path $eval_path --vocab_path $vocab_path --device $device --model_save_dir $model_save_dir --model_name $model_name

# Test
# test_path=/home/sunmengxuan/projects/disease_prediction/CDTPU/data/raw_eicu/eicu_data_v2/split_by_disease/test_6.json
# CUDA_VISIBLE_DEVICES=$device python main_seq.py --test --db_name $db_name --log_dir $log_dir --seed $seed --batch_size $batch_size --lr $lr --test_path $test_path --vocab_path $vocab_path --device $device --model_load_path $model_load_path --model_name $model_name