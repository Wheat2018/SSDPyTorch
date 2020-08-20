net_cfg = {
    'net_name': 'XFace_dualpath',
    'out_featmaps': False,
    # 'backbone': 'ResNet18FPN',
    'num_classes': 2,
    'finetune': False
}

anchor_cfg = {
    'dense_anchor': False,
    'feature_maps': [[128, 128], [64, 64], [32, 32]],
    'min_dim': 1024,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

anchor_cfg_s0 = {
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

anchor_cfg_s1 = {
    'dense_anchor': False,
    'feature_maps': [[128, 128], [64, 64], [32, 32]],
    'min_dim': 1024,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

anchor_cfg_s2 = {
    'dense_anchor': False,
    'feature_maps': [[160, 160], [80, 80], [40, 40]],
    'min_dim': 1280,
    'steps': [8, 16, 32],
    'min_sizes': [[16, 32], [64, 128], [196, 320]],
    'anchors': [16, 32, 64, 128, 196, 320],
    'aspect_ratios': [[], [], []],
    'variance': [0.1, 0.2],
    'clip': False
}

train_cfg = {
    'input_size': [640, 1024, 1280],
    'loss_type': 'focal',
    'anchor_base_loc_weight': 2.0,
    'anchor_base_cls_weight': 2.0,
    'anchor_free_loc_weight': 1.0,
    'anchor_free_cls_weight': 0.25,
    'ctr_weight': 1.0,
    'landmark_weight': 0.1,
    'distillation_weight': 3.0,
    'use_landmark': False,
    'aug_type': 'FaceBoxes',
    'lr_steps': [65, 85],  # step epoch for learning rate decreasing
    'save_folder': './weights/xface_dualpath_shuffle_multiscale',
}

test_cfg = {
    'save_folder': 'xface_dualpath_shuffle_multiscale',
    'is_anchor_base': True,
    'is_ctr': False
    
}


