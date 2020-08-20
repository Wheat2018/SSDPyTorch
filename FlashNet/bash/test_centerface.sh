#!/usr/bin/env bash
srun --partition=sensevideo -w SH-IDC1-10-5-38-94 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=dualpath_shuffle --kill-on-bad-exit=1 --cpus-per-task=1 \
python test_centerface.py --cfg_file ./configs/centerface.py --trained_model ./weights/centerface/AdamW/CenterFace_epoch_295.pth