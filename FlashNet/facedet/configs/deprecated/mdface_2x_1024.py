net_cfg = {
    'net_name': 'MDFace_2x',
    'out_featmaps': False,
    # 'backbone': 'ResNet18FPN',
    'num_classes': 2
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

train_cfg = {
    'input_size': 1024,
    'loss_type': 'focal',
    'loc_weight': 2.0,
    'cls_weight': 2.0,
    'landmark_weight': 0.1,
    'distillation_weight': 3.0,
    'use_landmark': False,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/MDFace_2x_1024',
}

test_cfg = {
    'save_folder': 'MDFace_2x',
    'is_anchor_base': True,
    'is_ctr': False

}