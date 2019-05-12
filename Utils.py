import errno
import librosa
import numpy as np
import os
import shutil
import Constants as c
from sklearn.utils import shuffle
from preprocess.sigproc_tools import lfilter

np.random.seed(12345)


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin):
    sin = lfilter([1, -1], [1, -0.99], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


def load_wav(filename, sample_rate=16000):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio


class CopyDataFiles(object):
    def __init__(self, data_origin_dir=None, data_temp_dir=None, n_samples=5):
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

            verif_test_file = self.data_origin_dir + 'verif_test.txt'
            metadata_file = self.data_origin_dir + 'vox1_meta.csv'
            test_paths_file = self.data_origin_dir + 'test_paths.txt'
            train_paths_file = self.data_origin_dir + 'train_paths.txt'

            shutil.copy(verif_test_file, self.data_temp_dir)
            shutil.copy(metadata_file, self.data_temp_dir)
            try:
                shutil.copy(test_paths_file, self.data_temp_dir)
                shutil.copy(train_paths_file, self.data_temp_dir)
            except IOError as e:
                # ENOENT(2): file does not exist, raised also on missing dest parent dir
                if e.errno != errno.ENOENT:
                    raise
                self._create_train_test_txt()

        self.move_from_origin_to_temp_dir(n_samples=n_samples)

    def move_from_origin_to_temp_dir(self, full_list=None, n_samples=3, from_path=None, to_path=None):
        """
        :param full_list: The List of paths in pairs or in single list
        :param n_samples: How many samples are gonna get copied to the to_path
        :param from_path: The path that the original data are located
        :param to_path: The path that the temp data is located (Default is data_temp_dir)
        :return: None
        """

        if from_path is None:
            from_path = self.data_origin_dir + 'wav/'
        if to_path is None:
            to_path = self.data_temp_dir

        if full_list is None:
            full_list = np.genfromtxt(to_path + 'train_paths.txt', dtype='str')

        full_list = shuffle(full_list)
        data_list = full_list[:n_samples]
        self._save_to_txt(self.data_temp_dir + 'samples_paths.txt', data_list)
        self.file_samples_paths = self.data_temp_dir + 'samples_paths.txt'
        if len(data_list.shape) > 1:
            # print(data_list)
            # if the list is actually a tuple
            data_list = data_list.flatten()

        for path in data_list:
            temp_from_path = os.path.join(from_path, path)
            temp_to_path = os.path.join(to_path, path)

            self._copy_and_overwrite(temp_from_path, temp_to_path)
        print(f'Moved {n_samples} files to {self.data_temp_dir}.\nFile format is: "speaker_id/video_id/file.wav"\n')

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

        return np.unique(files_1).reshape(-1, 1), labels, pairs

    def metadata_loader(self, path=None):
        if path is None:
            path = self.data_temp_dir

        data = np.genfromtxt(path + 'vox1_meta.csv', dtype='str')[1:, :]
        # print(np.unique(data[:,3]))
        return data


    def _copy_and_overwrite(self, from_path, to_path):
        try:
            shutil.copy(from_path, to_path)
        except IOError as e:
            # ENOENT(2): file does not exist, raised also on missing dest parent dir
            if e.errno != errno.ENOENT:
                raise
            # try creating parent directories
            os.makedirs(os.path.dirname(to_path))
            shutil.copy(from_path, to_path)

    def _create_train_test_txt(self):
        test_paths, *_ = self.verif_test_loader()
        print("No train and test files found. Preparing them now...")
        all_paths = []
        for root, dirs, files in os.walk(loader.data_origin_dir + 'wav'):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                path = os.path.join(root, file).replace(loader.data_origin_dir + 'wav/', '')
                all_paths.append(path)

        train_paths = []
        for all in all_paths:
            if not all in test_paths:
                train_paths.append(all)

        train_paths = np.array(train_paths).reshape(-1, 1)

        self._save_to_txt(self.data_temp_dir + 'train_paths.txt', train_paths)
        self._save_to_txt(self.data_temp_dir + 'test_paths.txt', test_paths)

        self.file_train_paths = self.data_temp_dir + 'train_paths.txt'
        self.file_test_paths = self.data_temp_dir + 'test_paths.txt'

    def _save_to_txt(self, path, data_list):
        np.savetxt(path, data_list, fmt='%s')


if __name__ == "__main__":
    # loader = CopyDataFiles()
    audio, sr = librosa.load('/Users/polaras/Documents/Useful/dataset/data_original/wav/id10309/_z_BR0ERa9g/00002.wav', sr=16000, mono=True)

