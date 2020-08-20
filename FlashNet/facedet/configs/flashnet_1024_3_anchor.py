net_cfg = {
    'net_name': 'FlashNet',
    'out_featmaps': False,
    'feat_adp': False,
    # 'backbone': 'ResNet18FPN',
    'num_classes': 2,
    'num_anchors_per_featmap': 3,
    'use_landmark': True
}

anchor_cfg = {
    'dense_anchor': False,
    'feature_maps': [[128, 128], [64, 64], [32, 32]],
    'min_dim': 1024,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 24, 32], [64, 96, 128], [196, 256, 320]],
    'anchors': [16, 24, 32, 64, 96, 128, 196, 256, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

train_cfg = {
    'input_size': 1024,
    'loss_type': 'focal',
    'loc_weight': 2.0,
    'cls_weight': 2.0,
    'landmark_weight': 0.01,
    'distillation_weight': 8.0,
    'mimic_weight': 50.0,
    'use_landmark': True,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/FlashNet_1024_3_anchor',
}

test_cfg = {
    'save_folder': 'FlashNet_1024_3_anchor',
    'is_anchor_base': True,
    'is_ctr': False

}