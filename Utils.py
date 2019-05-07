import errno
import numpy as np
import os
import shutil
import Constants as c
import librosa



def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio


def copy_and_overwrite(from_path, to_path):
    try:
        shutil.copy(from_path, to_path)
    except IOError as e:
        # ENOENT(2): file does not exist, raised also on missing dest parent dir
        if e.errno != errno.ENOENT:
            raise
        # try creating parent directories
        os.makedirs(os.path.dirname(to_path))
        shutil.copy(from_path, to_path)


class DataLoader(object):
    def __init__(self, data_origin_dir=None, data_temp_dir=None):
        """
        :param data_origin_dir: Path to original data. Contains everything with processing the files.
        :param data_temp_dir: Path to temp data used for experiments.
        """
        if data_origin_dir is None:
            data_origin_dir = c.DATA_ORIGIN
        if data_temp_dir is None:
            data_temp_dir = c.DATA_TEMP

        self.data_origin_dir = data_origin_dir
        self.data_temp_dir = data_temp_dir

        if not os.path.exists(self.data_temp_dir):
            os.mkdir(self.data_temp_dir)

            verif_test_dir = self.data_origin_dir + 'verif_test.txt'
            metadata_dir = self.data_origin_dir + 'vox1_meta.csv'

            shutil.copy(verif_test_dir, self.data_temp_dir)
            shutil.copy(metadata_dir, self.data_temp_dir)

    def move_from_origin_to_temp_dir(self, full_list, n_pairs=3, from_path=None, to_path=None):
        """
        :param full_list: The List of paths in pairs or in single list
        :param n_pairs: How many pair are gonna get copied to the to_path
        :param from_path: The path that the original data are located
        :param to_path: The path that the temp data is located (Default is data_temp_dir)
        :return: None
        """

        data_list = full_list[:n_pairs]

        if from_path is None:
            from_path = self.data_origin_dir + 'wav/'
        if to_path is None:
            to_path = self.data_temp_dir

        if len(data_list.shape) > 1:
            print(data_list)
            # if the list is actually a tuple
            data_list = data_list.flatten()

        for path in data_list:
            temp_from_path = os.path.join(from_path, path)
            temp_to_path = os.path.join(to_path, path)
            copy_and_overwrite(temp_from_path, temp_to_path)

    def verif_test_loader(self, path=None):
        """

        :param path: Path to verification file
        :return:
        """
        if path is None:
            path = self.data_temp_dir

        data = np.genfromtxt(path + 'verif_test.txt', dtype='str')

        labels = np.array(data[:, 0])

        files_1 = data[:, 1].reshape(-1, 1)
        files_2 = data[:, 2].reshape(-1, 1)
        pairs = np.concatenate((files_1, files_2), axis=1)

        return files_1, labels, pairs

    def metadata_loader(self, path=None):
        if path is None:
            path = self.data_temp_dir

        data = np.genfromtxt(path + 'vox1_meta.csv', dtype='str')[1:, :]
        # print(np.unique(data[:,3]))
        return data

    def fetc_data(self, path_to_data=None, list_data=None, n_samples=6):

        if list_data is None:
            labels, list_data = self.verif_test_loader(path_to_data)

        metadata = self.metadata_loader(path_to_data)
        self.move_from_origin_to_temp_dir(full_list=list_data, n_pairs=n_samples)


if __name__ == "__main__":
    loader = DataLoader()
    files, *_ = loader.verif_test_loader()
    print(files)
    print(len(files))
    print(np.unique(files))
    print(len(np.unique(files)))