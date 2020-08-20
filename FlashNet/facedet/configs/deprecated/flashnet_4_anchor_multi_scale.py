net_cfg = {
    'net_name': 'FlashNet',
    'out_featmaps': False,
    'feat_adp': False,
    # 'backbone': 'ResNet18FPN',
    'num_classes': 2,
    'num_anchors_per_featmap': 2,
    'use_landmark': True
}

anchor_cfg = {
    'dense_anchor': False,
    'feature_maps': [[40, 40], [20, 20], [10, 10]],
    'min_dim': 320,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}


anchor_cfg_s0 = {
    'dense_anchor': False,
    'feature_maps': [[40, 40], [20, 20], [10, 10]],
    'min_dim': 320,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

anchor_cfg_s1 = {
    'dense_anchor': False,
    'feature_maps': [[80, 80], [40, 40], [20, 20]],
    'min_dim': 640,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

train_cfg = {
    'input_size': [320, 640],
    'loss_type': 'focal',
    'loc_weight': 1.0,
    'cls_weight': 1.0,
    'landmark_weight': 0.01,
    'distillation_weight': 8.0,
    'mimic_weight': 50.0,
    'use_landmark': True,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/FlashNet_1024_2_anchor_multi_scale',
}

test_cfg = {
    'save_folder': 'FlashNet_1024_2_anchor_multi_scale',
    'is_anchor_base': True,
    'is_ctr': False

}