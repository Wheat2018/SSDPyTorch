net_cfg = {
    'net_name': 'CenterFace',
    'num_classes': 1,
    'use_ldmk': True,
    'ldmk_reg_type': 'heatmap'
}

train_cfg = {
    'input_size': 1024,
    'loss_type': 'focal',
    'wh_weight': 0.1,
    'cls_weight': 1.0,
    'ctr_weight': 0.1,
    'ldmk_weight': 1.0,
    'use_landmark': True,
    'aug_type': 'FaceBoxes',
    'lr_steps': [200, 250],  # step epoch for learning rate decreasing
    'save_folder': './weights/centerface_ldmk_heatmap',
}

test_cfg = {
    'save_folder': 'centerface_ldmk_heatmap',
    'is_ctr': False
    
}