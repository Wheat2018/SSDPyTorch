#!/usr/bin/env bash
python facedet/apis/trainers/train_anchor_base.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --ngpu 1 --optimizer AdamW \
--cfg_file ./configs/flashnet_1024.py -b 4

# srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
# --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=10 \
# python train_anchor_base.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --ngpu 1 --optimizer AdamW \
# --cfg_file ./configs/flashnet_1024.py -b 24

# srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
# --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=2 \
# python train_anchor_base.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --ngpu 1 --optimizer AdamW \
# --cfg_file ./configs/flashnet_1024.py -b 24


# srun --partition=TITANXP -w SZ-IDC1-172-20-20-37 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
# --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=2 \
# python train_anchor_base.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --ngpu 1 --optimizer AdamW \
# --cfg_file ./configs/flashnet_1024_3_anchor.py -b 24



# srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
# --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=2 \
# python train_anchor_base.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --ngpu 1 --optimizer AdamW \
# --cfg_file ./configs/flashnet_1024_3_anchor.py -b 2 --num_workers 0


# srun --partition=TITANXP -w SZ-IDC1-172-20-20-92 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=2 \
# python train_anchor_base_multi_scale.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --optimizer AdamW \
# --ngpu 1 --cfg_file ./configs/flashnet_1024_2_anchor_multi_scale.py -b 32 --resume_net ./weights/RetinaNet/Final_MobileFace.pth  --gpu_ids 7

# srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=flash --kill-on-bad-exit=1 --cpus-per-task=2 \
# python train_anchor_base_multi_scale.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --optimizer AdamW \
# --ngpu 1 --cfg_file ./configs/flashnet_1024_2_anchor_multi_scale.py -b 32 --resume_net ./weights/RetinaNet/Final_MobileFace.pth  --gpu_ids 7


# srun --partition=TITANXP -w SZ-IDC1-172-20-20-37 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=flash --kill-on-bad-exit=1 \
# python train_anchor_base_multi_scale.py --training_dataset /mnt/lustre/geyongtao/dataset/WIDER_ldmk --optimizer AdamW --ngpu 1 \
# --cfg_file ./configs/flashnet_1024_2_anchor_multi_scale.py -b 32 \
# --resume_net ./weights/FlashNet_1024_2_anchor_multi_scale/AdamW/epoch_40.pth --resume_epoch 40  --gpu_ids 7 -b 24 --lr 0.0001