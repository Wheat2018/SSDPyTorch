'''
python -m tvm.exec.rpc_tracker
python -m tvm.exec.rpc_server --tracker=172.20.106.93:9190 --key=android

python3 -m tvm.exec.query_rpc_tracker #check

python speed_benmark/tvm_arm_cpu_benchmark.py --host 172.20.106.93 --device p20


'''
import onnx
import torch
import time
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib.util import tempdir
from PIL import Image
from mmcv import Config
import argparse
import models
from tvm.contrib import graph_runtime as runtime
import threading

parser = argparse.ArgumentParser(description='MobileFace Training')
parser.add_argument('--cfg_file', default='./configs/mdface_light_1024.py', type=str,
                    help='model config file')
parser.add_argument("--repeat", type=int, default=20)
parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
parser.add_argument("--device", type=str, choices=
                    ['rk3399', 'mate10', 'mate10pro', 'p20', 'p20pro',
                    'pixel2', 'rasp3b', 'pynq'], default='p20',
                    help="The model of the test device. If your device is not listed in "
                         "the choices list, pick the most similar one as argument.")
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=9190)
parser.add_argument("--rpc_key", type=str, default="p20")
parser.add_argument("--resolution", type=int, nargs=2,
                    help="The resolution of input image")
args = parser.parse_args()

cfg = Config.fromfile(args.cfg_file)

def transform_image(image):                # 定义转化函数，将PIL格式的图像转化为格式维度的numpy格式数组
    mean = [123., 117., 104.]  # 在ImageNet上训练数据集的mean和std
    std = [58.395, 57.12, 57.375]
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image


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


def benchmark(onnx_model, target, repeat=args.repeat):
    # connect to remote device
    network = cfg['net_cfg']['net_name']
    tracker = tvm.rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(args.rpc_key)

    # target = "llvm" or
    input_name = 'img'  # 注意这里为之前导出onnx模型中的模型的输入id，这里为0
    shape_dict = {input_name: x.shape}
    # 利用Relay中的onnx前端读取我们导出的onnx模型
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(mod, target, params=params)

    tmp = tempdir()
    if 'android' in str(target):
        from tvm.contrib import ndk
        filename = "%s.so" % network
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "%s.tar" % network
        lib.export_library(tmp.relpath(filename))

    # upload library and params
    print("%-20s uploading..." % network)
    print(target)
    print(filename)
    ctx = remote.context(str(target), 0)

    remote.upload(tmp.relpath(filename))
    import pdb
    pdb.set_trace()

    rlib = remote.load_module(filename)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array(x.astype('float32'))
    module.set_input('0', data_tvm)
    module.set_input(**params)
    # evaluate
    print("%-20s evaluating..." % network)
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))



model = models.__dict__[cfg['net_cfg']['net_name']](phase='test', cfg=cfg['net_cfg'])
# state_dict = torch.load('Final_MobileFace.pth', map_location='cpu') # add map_location='cpu' if no gpu
# model.load_state_dict(state_dict)
# model = load_model(model, 'Final_MobileFace.pth')
# 导出 onnx 模型
example = torch.rand(1, 3, args.resolution[0], args.resolution[1])   # 假想输入
torch_out = torch.onnx.export(model,
                              example,
                              cfg['net_cfg']['net_name']+".onnx",
                              verbose=True,
                              export_params=True,   # 不带参数输出
                              opset_version=9,
                              input_names=['img']
                              )

onnx_model = onnx.load(cfg['net_cfg']['net_name']+".onnx")  # 导入模型
# onnx_model = onnx.load('faceboxes.onnx')  # 导入模型

img = Image.open('./speed_benchmark/4_Dancing_Dancing_4_690.jpg').resize((args.resolution[0], args.resolution[1]))  #这里我们将图像resize为特定大小
x = transform_image(img)

target = tvm.target.arm_cpu(args.device)

if args.thread == 1:
    benchmark(onnx_model, target)
else:
    threads = list()
    for n in range(args.thread):
        thread = threading.Thread(target=benchmark, args=([onnx_model, target]), name="thread%d" % n)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
