#!/usr/bin/env bash
srun --partition=TITANXP -w SZ-IDC1-172-20-20-35 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=3 \
python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface_ldmk_hp.py -b 32 --ngpu 1 --optimizer AdamW

srun --partition=TITANXP -w SZ-IDC1-172-20-20-35 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=coord --kill-on-bad-exit=1 --cpus-per-task=3 \
python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface_ldmk_coord.py --ngpu 1 --optimizer AdamW -b 32




srun --partition=TITANXP --mpi=pmi2 --gres=gpu:0 -n1 --cpus-per-task 1 --ntasks-per-node=1 --job-name=mt --kill-on-bad-exit=1 -w SZ-IDC1-172-20-20-35 \
python train_centerface_with_ldmk.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/centerface_ldmk_hp.py -b 3 --ngpu 1 --optimizer AdamW --num_workers 0



srun --partition=TITANXP --mpi=pmi2 --gres=gpu:0 -n1 --cpus-per-task 1 --ntasks-per-node=1 --job-name=mt --kill-on-bad-exit=1 -w SZ-IDC1-172-20-20-92 \
python vis_kpface.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk \
--cfg_file ./configs/kpface.py -b 1 --ngpu 1 --resume_net