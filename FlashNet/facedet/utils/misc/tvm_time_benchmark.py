import onnx
import torch
import time
import tvm
import numpy as np
import tvm.relay as relay
from PIL import Image
from mobileface_fpn import MobileFace_FPN

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# model = MobileFace_FPN(phase='test')    # initialize detector
# state_dict = torch.load('Final_MobileFace.pth', map_location='cpu') # add map_location='cpu' if no gpu
# model.load_state_dict(state_dict)
# model = load_model(model, 'Final_MobileFace.pth')
# 导出 onnx 模型
example = torch.rand(1, 3, 640, 640)   # 假想输入
example = torch.rand(1, 3, 320, 320)   # 假想输入
torch_out = torch.onnx.export(model,
                              example,
                              "mobileface_fpn.onnx",
                              verbose=True,
                              export_params=True   # 带参数输出
                              )


# onnx_model = onnx.load('MobileNetV2-SSDLite-Hand-9496_1.onnx')  # 导入模型
onnx_model = onnx.load('mobileface_fpn.onnx')  # 导入模型
# onnx_model = onnx.load('faceboxes.onnx')  # 导入模型

mean = [123., 117., 104.]                   # 在ImageNet上训练数据集的mean和std
std = [58.395, 57.12, 57.375]


def transform_image(image):                # 定义转化函数，将PIL格式的图像转化为格式维度的numpy格式数组
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image

img = Image.open('tvm_plane.jpg').resize((640, 640)) # 这里我们将图像resize为特定大小
# img = Image.open('tvm_plane.jpg').resize((320, 320)) # 这里我们将图像resize为特定大小
x = transform_image(img)
target = 'llvm'

# import pdb
# pdb.set_trace()
input_name = '0'  # 注意这里为之前导出onnx模型中的模型的输入id，这里为0
shape_dict = {input_name: x.shape}
# 利用Relay中的onnx前端读取我们导出的onnx模型
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', mod, tvm.cpu(0), target)

dtype = 'float32'
func = intrp.evaluate()

output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
import pdb
pdb.set_trace()
print(output.argmax())

since = time.time()
for i in range(1000):
    output = func(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
time_elapsed = time.time() - since
print('Time elapsed is {:.0f}m {:.0f}s'.
      format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间