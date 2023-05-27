for data_num in 0 1 2 3 4 5 6 7 8 9 10 11
do
    python3 main.py\
    --gpu_id 3\
    --data_num $data_num\
    --seq_length 576\
    --dataset ASD\
    --input_c 19\
    --output_c 19\
    --anormly_ratio 5\
    --batch_size 4\
    | tee -a ./logs/ASD_time_all.out
done

for data_num in 0 1 2 3 4 5 6 7 8 9 10 11
do
    python3 main_freq.py\
    --gpu_id 3\
    --data_num $data_num\
    --seq_length 576\
    --nest_length 288\
    --dataset ASD\
    --data_loader load_ASD\
    --input_c 19\
    --output_c 19\
    --anormly_ratio 5\
    --batch_size 4\
    | tee -a ./logs/ASD_freq_all.out
done

for data_num in 0 1 2 3 4 5 6 7 8 9 10 11
do
    python3 evaluation.py\
    --data_num $data_num\
    --dataset ASD\
    --data_loader load_ASD\
    --seq_length 576\
    | tee -a ./logs/ASD_evaluation.out
done