import os
import constants as c
import shutil
from tqdm import tqdm

temp_data_dir = '/Users/polaras/Documents/Useful/dataset/temp_data'


def move_files_for_upload():

    with open('paths.txt') as f:
        files_paths = f.read().splitlines()

    for root, dirs, files in tqdm(os.walk(c.DATA_ORIGIN[:-14])):
        for i, file in enumerate(files):
            full_path = os.path.join(root, file)

            if not file.endswith(".wav") or full_path[40:] in files_paths:
                continue

            if not os.path.exists(c.DATA_ORIGIN + 'temp_data/'):
                os.mkdir(c.DATA_ORIGIN + 'temp_data/')

            src = os.path.join(root, file)
            dst = src.replace('data_original', 'temp_data')

            dist_dir = os.path.dirname(dst)

            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)

            shutil.copyfile(src, dst)


if __name__ == '__main__':
    move_files_for_upload()