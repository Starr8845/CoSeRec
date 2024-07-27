python src/main.py --data_name Sports_and_Outdoors --cl --model_idx cl_freq_emb
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx freq_emb
wait
python src/main.py --data_name Sports_and_Outdoors --model_idx freq_emb_multi_neg --multi_neg
wait 
python src/main.py --data_name Sports_and_Outdoors --cl --model_idx cl_freq_emb_multi_neg --multi_neg
wait



