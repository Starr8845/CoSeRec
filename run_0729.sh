python src/main.py --data_name Sports_and_Outdoors --model_idx single_1024_2 --batch_size 1024
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_mid_low_1024 --batch_size 1024 --cl --base_augment_type mask --mask_strategy mask_mid_low_all
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.5_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.5
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.1_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.1
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.3_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.3
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_mid_low_1024 --batch_size 1024 --cl --base_augment_type mask --mask_strategy mask_mid_low
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.9_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.9
wait
