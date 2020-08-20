#!/usr/bin/env bash
srun --partition=SenseVideo5 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python test_xface_dualpath_single.py --cfg_file ./configs/xface_dualpath.py --trained_model ./weights/xface_dualpath/AdamWFinal_XFace.pth
srun --partition=SenseVideo5 -w SH-IDC1-10-5-40-189 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python test_xface_dualpath.py --cfg_file ./configs/xface_dualpath.py --trained_model ./weights/xface_dualpath/AdamWFinal_XFace.pth


srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python test_xface_dualpath.py --cfg_file ./configs/xface_dualpath.py --trained_model ./weights/xface_dualpath/AdamWFinal_XFace.pth

srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python test_xface_dualpath.py --cfg_file ./configs/xface_dualpath_shuffle.py --trained_model ./weights/xface_dualpath_shuffle/AdamW/XFace_epoch_255.pth

srun --partition=TITANXP --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python test_single_img.py --cfg_file ./configs/xface_dualpath_shuffle.py --trained_model ./weights/xface_dualpath_shuffle/AdamW/XFace_epoch_255.pth --dataset Archive


srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=extrapolate --kill-on-bad-exit=1 \
python dataset/tools/celeba2voc_lmdk.py