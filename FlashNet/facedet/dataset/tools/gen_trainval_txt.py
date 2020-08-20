import os


def gen_trainval_txt(xml_dir, save_dir):
    names = os.listdir(xml_dir)
    # import pdb
    # pdb.set_trace()
    i = 0
    filename = os.path.join(save_dir, 'trainval.txt')
    f = open(filename, 'w')
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        f.write(name + '\n')
        i = i + 1
        print(i)


xml_dir = '/home/gyt/dataset/vehicle/Annotations'
save_dir = '/home/gyt/dataset/vehicle/ImageSets/Main'
gen_trainval_txt(xml_dir, save_dir)