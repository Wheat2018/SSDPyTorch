net_cfg = {
    'net_name': 'KPFace',
    'num_classes': 1,
    'use_ldmk': False
}

train_cfg = {
    'input_size': 640,
    'loss_type': 'focal',
    'box_weight': 1.0,
    'ldmk_weight': 5.0,
    'use_landmark': False,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/kpface',
}

test_cfg = {
    'save_folder': 'kpface',
    'is_anchor_base': True,
    'is_ctr': False
    
}