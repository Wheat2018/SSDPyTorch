#!/usr/bin/env bash

srun --partition=SenseVideo5 -w SH-IDC1-10-5-40-188 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath --kill-on-bad-exit=1 --cpus-per-task=10 \
python train_xface_dualpath.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_train_5 -b 32 --ngpu 1 --optimizer AdamW


srun --partition=SenseVideo5 -w SH-IDC1-10-5-40-188 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_finetune --kill-on-bad-exit=1 --cpus-per-task=10 \
python train_xface_dualpath_finetune.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_train_320 -b 8 --ngpu 1 --optimizer AdamW

srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_finetune --kill-on-bad-exit=1 --cpus-per-task=10 \
python train_xface_dualpath_finetune1.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_train_320 -b 8 --ngpu 1 --optimizer AdamW \
--resume_net ./weight/


srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=10 \
python train_xface_dualpath.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_train_5 \
--cfg_file ./configs/xface_dualpath_shuffle.py -b 32 --ngpu 1 --optimizer AdamW


srun --partition=sensevideo -w SH-IDC1-10-5-38-94 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=3 \
python train_xface_dualpath.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_train_5 \
--cfg_file ./configs/xface_dualpath_shuffle.py -b 32 --ngpu 1 --optimizer AdamW


srun --partition=sensevideo -w SH-IDC1-10-5-38-93 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=3 \
CUDA_LAUNCH_BLOCKING=1 python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface.py -b 1 --ngpu 1 --optimizer AdamW

srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=3 \
python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface.py -b 2 --ngpu 1 --optimizer AdamW

srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=3 \
python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface_ldmk.py -b 2 --ngpu 1 --optimizer AdamW --num_workers 0



srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 \
python models/centerface.py


srun --partition=sensevideo -w SH-IDC1-10-5-38-102 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 \
python models/centerface.py