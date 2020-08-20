#!/usr/bin/env bash
srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python eval_aflw.py --cfg_file ./configs/flashnet_1024_2_anchor_multi_scale.py \
--trained_model ./weights/FlashNet_1024_2_anchor_multi_scale/AdamW/epoch_4.pth --size_img 256

srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python eval_aflw.py --cfg_file ./configs/flashnet_1024_2_anchor_multi_scale.py \
--trained_model ./weights/FlashNet_1024_2_anchor_multi_scale/AdamW/epoch_35.pth --size_img 256


srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python dataset/tools/aflw2voc_lmdk.py