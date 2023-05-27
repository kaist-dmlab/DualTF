for data_num in 0 1 2 3 4 5 6 7 8
do
    python3 main.py\
    --gpu_id 6\
    --data_num $data_num\
    --seq_length 286\
    --dataset ECG\
    --input_c 2\
    --output_c 2\
    --anormly_ratio 0.1\
    --batch_size 4\
    | tee -a ./logs001/ECG_time_all.out
done

for data_num in 0 1 2 3 4 5 6 7 8
do
    python3 main_freq.py\
    --gpu_id 6\
    --data_num $data_num\
    --seq_length 286\
    --nest_length 143\
    --dataset ECG\
    --data_loader load_ECG\
    --input_c 2\
    --output_c 2\
    --anormly_ratio 0.1\
    --batch_size 4\
    | tee -a ./logs001/ECG_freq_all.out
done

for data_num in 0 1 2 3 4 5 6 7 8
do
    python3 evaluation.py\
    --data_num $data_num\
    --dataset ECG\
    --data_loader load_ECG\
    --seq_length 286\
    | tee -a ./logs001/ECG_evaluation.out
done