net_cfg = {
    'net_name': 'CenterFace',
    'num_classes': 1,
    'use_ldmk': True,
    'ldmk_reg_type': 'coord'
}

train_cfg = {
    'input_size': 640,
    'loss_type': 'focal',
    'wh_weight': 0.1,
    'cls_weight': 1.0,
    'ctr_weight': 0.1,
    'ldmk_weight': 1.0,
    'use_landmark': True,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/centerface_ldmk_coord',
}

test_cfg = {
    'save_folder': 'centerface_ldmk_coord',
    'is_anchor_base': True,
    'is_ctr': False
    
}