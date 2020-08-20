import os
import random

def gen_trainval_txt(xml_dir, save_dir):
    # names = os.listdir(xml_dir)
    # import pdb
    # pdb.set_trace()
    f = open("/Users/geyongtao/Documents/dataset/WIDER_train_5/ImageSets/Main/trainval.txt", "r")
    names = f.readlines()
    num_samples = len(names)
    random_idx = random.sample(range(num_samples), num_samples)

    shuffle_names = []
    for idx in random_idx:
        shuffle_names.append(names[idx])

    i = 0
    filename = os.path.join(save_dir, 'trainval_shuffle.txt')
    f = open(filename, 'w')
    for name in shuffle_names:
        index = name.rfind('.')
        name = name[:index]
        f.write(name + '\n')
        i = i + 1
        print(i)

xml_dir = '/Users/geyongtao/Documents/dataset/WIDER_train_5/Annotations'
save_dir = '/Users/geyongtao/Documents/dataset/WIDER_train_5/ImageSets/Main'
gen_trainval_txt(xml_dir, save_dir)