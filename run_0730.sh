python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024_only_aug_high --batch_size 1024 --cl --aug_by_target high
wait 
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024_only_aug_mid_low --batch_size 1024 --cl --aug_by_target mid_low
wait 
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024_contrastanchor --batch_size 1024 --cl --contrast_anchor
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_1024_only_aug_low --batch_size 1024 --cl --aug_by_target low
wait 
python src/main.py --data_name Sports_and_Outdoors --model_idx single_cl_mask_1024_contrastanchor --batch_size 1024 --cl --base_augment_type mask --contrast_anchor
wait




python src/main.py --data_name Sports_and_Outdoors --model_idx single_1024_only_aug_wo_infonce --batch_size 1024 --cl --cl_only_aug
