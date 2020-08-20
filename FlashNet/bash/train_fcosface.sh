#!/usr/bin/env bash
srun --partition=Test --mpi=pmi2 --gres=gpu:0 -n1 --cpus-per-task 2 --ntasks-per-node=1 --job-name=mt --kill-on-bad-exit=1 \
python train_fcosface.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --cfg_file ./configs/fcosface.py --ngpu 1 --optimizer AdamW -b 3 --num_workers 0

srun --partition=TITANXP --mpi=pmi2 --gres=gpu:0 -n1 --cpus-per-task 4 --ntasks-per-node=1 --job-name=mt --kill-on-bad-exit=1 -w SZ-IDC1-172-20-20-38 \
python train_fcosface.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --cfg_file ./configs/fcosface.py --ngpu 2 --optimizer AdamW -b 64