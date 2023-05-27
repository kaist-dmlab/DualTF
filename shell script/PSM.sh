for data_num in 0
do
    python3 main.py\
    --gpu_id 3\
    --form $data_num\
    --seq_length 720\
    --dataset PSM\
    --input_c 25\
    --output_c 25\
    --batch_size 4\
    | tee -a ./logs/PSM_time_all.out
done

for data_num in 0
do
    python3 main_freq.py\
    --gpu_id 3\
    --data_num $data_num\
    --seq_length 720\
    --nest_length 360\
    --dataset PSM\
    --data_loader load_PSM\
    --input_c 25\
    --output_c 25\
    --anormly_ratio 27\
    --batch_size 4\
    | tee -a ./logs/PSM_freq_all.out
done

for data_num in 0
do
    python3 evaluation.py\
    --data_num $data_num\
    --dataset PSM\
    --data_loader load_PSM\
    --seq_length 720\
    | tee -a ./logs/PSM_evaluation.out
done