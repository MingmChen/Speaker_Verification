import errno

import numpy as np
import os
import shutil
import Constants as c
data_temp_dir = os.path.join(c.ROOT, 'data_temp')



def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def verif_loader(path=None):
    if path is None:
        path = data_temp_dir

    data = np.genfromtxt(path + '/verif_test.txt', dtype='str')
    labels = np.array(data[:, 0])

    files_1 = data[:, 1].reshape(-1, 1)
    files_2 = data[:, 2].reshape(-1, 1)
    pairs = np.concatenate((files_1, files_2), axis=1)

    return labels, pairs


def metadata_loader(path=None):
    if path is None:
        path = data_temp_dir

    data = np.genfromtxt(path + '/vox1_meta.csv', dtype='str')[1:, :]
    return data


def move_from_origin_to_temp(full_path_list, number_of_pairs=3, inpath=None, outpath=data_temp_dir, ):
    """
    :param pairs: The List of pairs
    :param number_of_pairs: How many pair are gonna get copied to the outpath
    :param inpath: The path that the original data are located
    :param outpath: The path that the temp data is located (Default is data_temp_dir)
    :return: None
    """
    full_path_list = full_path_list[:number_of_pairs]

    if inpath is None:
        inpath = '/Users/polaras/Documents/Useful/dataset/data_original/wav/'

    if len(full_path_list.shape) > 1:  # if the list is actually a tuple
        full_path_list = full_path_list.flatten()

    for path in full_path_list:
        temp_inpath = os.path.join(inpath, path)
        temp_outpath = os.path.join(outpath, path)
        copy(temp_inpath,temp_outpath)


if __name__ == "__main__":
    labels, pairs = verif_loader()
    move_from_origin_to_temp(pairs)
