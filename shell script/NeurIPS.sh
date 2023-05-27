for form in shaplet seasonal trend point context
do
    python3 main.py\
    --gpu_id 6\
    --form $form\
    --seq_length 50\
    --dataset NeurIPSTS\
    --batch_size 4\
    | tee -a ./logs/NeurIPS_time_all.out
done

for form in shaplet seasonal trend point context
do
    python3 main_freq.py\
    --gpu_id 4\
    --form $form\
    --seq_length 50\
    --nest_length 25\
    --dataset NeurIPSTS\
    --data_loader load_tods\
    --batch_size 4\
    | tee -a ./logs/NeurIPS_freq_all.out
done

for form in shaplet seasonal trend point context
do
    python3 evaluation.py\
    --form $form\
    --dataset NeurIPSTS\
    --data_loader load_tods\
    --seq_length 50\
    | tee -a ./logs/NeurIPS_evaluation.out
done