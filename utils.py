import errno
import librosa
import numpy as np
import os
import shutil
import speech_feature_extraction.speechpy as speech
import constants as c
from sklearn.utils import shuffle
import torch.optim as optim
import torch
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

np.random.seed(12345)

# with equal number of samples
def create_N_first_train_paths(N):

    train_paths_origin = np.genfromtxt(c.DATA_ORIGIN + 'train_paths.txt', dtype='str')
    path_list = []

    train_ids = []
    for index, train in enumerate(train_paths_origin):
        train_id = train[0:7]
        train_ids.append(train_id)
        path_list.append(train)
        # print(train)

    uniques = np.unique(train_ids)
    uni_500 = uniques[0:N]
    final_trains = []

    min_length = 100000
    for id in uni_500:

        ids = [v for i, v in enumerate(path_list) if id in v]
        if len(ids) < min_length:
            min_length = len(ids)

        # final_trains.extend(ids)

    for id in uni_500:

        ids = [v for i, v in enumerate(path_list) if id in v]
        iddds = ids[0:min_length]
        print(iddds)
        final_trains.extend(iddds)


    np.savetxt('{}_first_ids.txt'.format(N), final_trains, fmt='%s')
    create_masked_indices('{}_first_ids.txt'.format(N))


def create_masked_indices(file=None):
    if file is None:
        train_paths_origin = np.genfromtxt(c.DATA_ORIGIN + 'train_paths.txt', dtype='str')
    else:
        train_paths_origin = np.genfromtxt(c.ROOT +'/' + file, dtype='str')

    indexed_labels = {}
    path_list = []
    for index, train in enumerate(train_paths_origin):
        train = train[0:7]
        path_list.append(train)
        # print(train)

    uniques = np.unique(list(path_list))

    for index, train in enumerate(uniques):
        indexed_labels[train] = index

    if file is None:
        np.save('labeled_indices', indexed_labels)
    else:
        np.save(file[:-4], indexed_labels)


def split_sets(dataset, validation_split=c.VALIDATION_SPLIT, shuffle_dataset=c.SHUFFLE_DATA, batch_size=c.BATCH_SIZE):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader


def LossAndOptimizer(learning_rate, model):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss, optimizer


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    print('saving model ...')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, c.MODEL_DIR + '/model_best.pt')


def calculate_accuracy(model, X, Y):
    oupt = model(X)
    (max_vals, arg_maxs) = torch.max(oupt.data, dim=1)

    num_correct = torch.sum(Y == arg_maxs)
    acc = (num_correct * 100.0 / len(Y))
    return acc.item()


def plot_loss_acc(train_loss, train_acc):
    fig = plt.figure()
    ax = fig.gca()
    plt.title('Loss')
    plt.plot(train_loss, color='r')
    plt.xlabel('Epochs')
    plt.legend(['train_cost'])
    plt.grid()
    plt.savefig('final_loss.png')

    fig = plt.figure()
    ax = fig.gca()
    plt.title('Accuracy')
    plt.plot(train_acc, color='r')
    plt.xlabel('Epochs')
    plt.legend(['train_accuracy'])
    plt.grid()
    plt.savefig('final_accuracy.png')


def save_file(path_to_file, file):
    torch.save(file, path_to_file)


def load_file(path_to_file):
    file = torch.load(path_to_file)
    return file


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


def load_wav(filename, sample_rate=c.SAMPLE_RATE):
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

    def _verif_test_loader(self, path=None):
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

    def _metadata_loader(self, path=None):
        if path is None:
            path = self.data_temp_dir

        data = np.genfromtxt(path + 'vox1_meta.csv', dtype='str')[1:, :]
        # print(np.unique(data[:,3]))
        return data

    def _copy_and_overwrite(self, from_path, to_path):
        """

        :param from_path: Source path to copy
        :param to_path: Destination path
        :return: None
        """
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
        test_paths, *_ = self._verif_test_loader()
        print("No train and test files found. Preparing them now...")
        all_paths = []
        for root, dirs, files in os.walk(self.data_origin_dir + 'wav'):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                path = os.path.join(root, file).replace(self.data_origin_dir + 'wav/', '')
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


class ToTensor(object):
    """Return the output in tensor format.
    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        return feature, label


class FeatureCube(object):
    """Return a feature cube of desired size.
    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape):
        assert isinstance(cube_shape, (tuple))
        self.cube_shape = cube_shape
        self.num_frames = cube_shape[0]
        self.num_coefficient = cube_shape[1]
        self.num_utterances = cube_shape[2]

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        # Feature cube.
        feature_cube = np.zeros((self.num_utterances, self.num_frames, self.num_coefficient), dtype=np.float32)

        # Get some random starting point for creation of the future cube of size (num_frames x num_coefficient x num_utterances)
        # Since we are doing random indexing, the data augmentation is done as well because in each iteration it returns another indexing!
        idx = np.random.randint(feature.shape[0] - self.num_frames, size=self.num_utterances)

        for num, index in enumerate(idx):
            feature_cube[num, :, :] = feature[index:index + self.num_frames, :]

        # return {'feature': feature_cube, 'label': label}
        return {'feature': feature_cube[None, :, :, :], 'label': label}


class CMVN(object):
    """Cepstral mean variance normalization."""

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        feature = speech.processing.cmvn(feature, variance_normalization=True)
        return {'feature': feature, 'label': label}


if __name__ == "__main__":
    # loader = CopyDataFiles(n_samples=10)
    # file = np.genfromtxt(c.DATA_TEMP + 'samples_paths.txt', dtype='str')
    #
    # file = sorted(file)
    create_N_first_train_paths(500)
