python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024 --batch_size 1024 --cl
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_1024 --batch_size 1024 --cl --base_augment_type mask
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_crop_1024 --batch_size 1024 --cl --base_augment_type crop
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_substitute_1024 --batch_size 1024 --cl --base_augment_type substitute
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_insert_1024 --batch_size 1024 --cl --base_augment_type insert
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_reorder_1024 --batch_size 1024 --cl --base_augment_type reorder



python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_high_1024 --batch_size 1024 --cl --base_augment_type mask --mask_strategy mask_high


python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_mid_low_1024 --batch_size 1024 --cl --base_augment_type mask --mask_strategy mask_mid_low


python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_mid_low_1024 --batch_size 1024 --cl --base_augment_type mask --mask_strategy mask_mid_low_all


python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.1_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.1

python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.3_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.3

python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.5_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.5

python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_0.9_1024 --batch_size 1024 --cl --base_augment_type mask --gamma 0.9





python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024_freq --batch_size 1024 --cl --use_freq

python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_crop_1024_freq --batch_size 1024 --cl --base_augment_type crop --use_freq



