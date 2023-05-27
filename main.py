import argparse
import os

from dualTF import TimeReconstructor
import torch
from torch.backends import cudnn

def main(opts):
    cudnn.benchmark = True
    # check settings
    print(f"GPU_ID: {opts.gpu_id}")

    framework = TimeReconstructor(vars(opts))
    framework.progress()
    return framework

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings for Daul-TF (Time)')

    # hardware settings
    parser.add_argument('--gpu_id', default='4', type=str, help='gpu_ids: e.g. 0, 1, 2, 3, 4, 5, 6')

    # model settings
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='NeurIPSTS')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--form', type=str, default='seasonal')
    parser.add_argument('--data_num', type=int, default=0)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1)

    opts = parser.parse_args()

    # set gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES']= opts.gpu_id

    args = vars(opts)
    print('############ Arguments ############')
    print(args)
    
    print('############ Print Option Items ############')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('############################################')
    main(opts)