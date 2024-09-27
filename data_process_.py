import numpy as np
import os
from slc_functions import gen_spectrogram_2
import argparse
def gen_all_spec(slc_root, spe_root, win):
    slc_list = os.listdir(slc_root)
    if not os.path.exists(spe_root):
        os.mkdir(spe_root)
    for cate in slc_list:
        if not os.path.exists(spe_root + cate):
            os.mkdir(spe_root + cate)

        data_list = os.listdir(slc_root + cate)
        for data in data_list:
            if not os.path.exists(spe_root + cate + '/' + data) and data[-3:] == 'npy':
                print(data)
                slc_data = np.load(slc_root + cate + '/' + data)
                spectrogram = np.log(1+np.abs(gen_spectrogram_2(slc_data, win)))

                np.save(spe_root + cate + '/' + data, spectrogram)







if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train_joint')

    parser.add_argument('--slc_root', type=str, default='/slc_data/')
    parser.add_argument('--spe_root', type=str, default='/spe_data/')
    parser.add_argument('--win', type=float, default=0.5)

    args = parser.parse_args()

    slc_root = args.slc_root
    spe4D_root = args.spe4D_root
    spe3D_root = args.spe3D_root
    win = args.win

    if slc_root:
        if spe4D_root:
            print('generate 4D signals...')
            gen_all_spec(slc_root, spe4D_root, win) # generate the 4-D signals
            ff=0

