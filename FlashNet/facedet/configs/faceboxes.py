net_cfg = {
    'net_name': 'FaceBoxes',
    # 'backbone': 'ResNet18FPN',
    'num_classes': 2
}

anchor_cfg = {
    'dense_anchor': True,
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'anchors': [32, 64, 128, 256, 512],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

train_cfg = {
    'input_size': 1024,
    'loss_type': 'focal',
    'loc_weight': 2.0,
    'cls_weight': 2.0,
    'landmark_weight': 0.1,
    'distillation_weight': 3.0,
    'use_landmark': False,
    'aug_type': 'FaceBoxes',
    'save_folder': './weights/FaceBoxes',
    'lr_steps': [200, 250]
}

test_cfg = {
    'save_folder': 'FaceBoxes',
    'is_anchor_base': True,
    'is_ctr': False
    
}