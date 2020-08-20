net_cfg = {
    'net_name': 'CenterFace',
    'num_classes': 1,
    'use_ldmk': False
}

train_cfg = {
    'input_size': 640,
    'loss_type': 'focal',
    'wh_weight': 0.5,
    'cls_weight': 1.0,
    'ctr_weight': 1.0,
    'use_landmark': False,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/centerface',
}

test_cfg = {
    'save_folder': 'centerface',
    'is_anchor_base': True,
    'is_ctr': False
    
}