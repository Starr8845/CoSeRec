python src/main.py --data_name Sports_and_Outdoors --model_idx single_base_256 --batch_size 256
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_base_1024 --batch_size 1024
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_base_256_freq --batch_size 256 --use_freq
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_base_1024_freq --batch_size 1024 --use_freq
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_base_1024_multi_neg --batch_size 1024 --multi_neg
wait



python src/main.py --data_name Sports_and_Outdoors --model_idx single_1024_dict --batch_size 1024

