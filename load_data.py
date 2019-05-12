import os
import Utils
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import speechpy
import Constants as c
import numpy as np


# https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/code/0-input/input_feature.py
class AudioDataset(data.Dataset):
    def __init__(self, files_path, audio_dir, transform=None):
        """
        :param files_path: Path to the .txt file which contains all the file_list
        :param audio_dir:  Directory with all the audio files.
        :param transform:  Optional transform to be applied
                on a sample.
        """
        self.audio_dir = audio_dir
        self.transform = transform

        # Open the .txt file and create a list from each line.
        content = np.genfromtxt(files_path, dtype='str')

        list_files = []
        for x in content:
            sound_file_path = os.path.join(self.audio_dir, x)
            # print(sound_file_path)
            try:
                file_size = os.path.getsize(sound_file_path)
                assert file_size > 1000, "Bad file!"
                # Add to list if file is OK!
                list_files.append(x)


            except OSError as err:
                print("OS error: {0}".format(err))

            except ValueError:
                print('file %s is corrupted!' % sound_file_path)

        # Save the correct files
        self.sound_files = list_files

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        sound_file_path = os.path.join(self.audio_dir, self.sound_files[idx])

        signal = Utils.load_wav(sound_file_path)

        frames = speechpy.processing.stack_frames(signal, sampling_frequency=c.SAMPLE_RATE, frame_length=0.025,
                                                  frame_stride=0.01,
                                                  zero_padding=True)

        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * c.NUM_COEF)[:, 1:]

        logenergy = speechpy.feature.lmfe(signal,
                                          sampling_frequency=c.SAMPLE_RATE,
                                          frame_length=c.FRAME_LEN,
                                          frame_stride=c.FRAME_STEP,
                                          num_filters=c.NUM_COEF,
                                          fft_length=c.NUM_FFT)

        # Label extraction
        label = self.sound_files[idx][:7]

        sample = {'feature': logenergy, 'label': label}

        # Apply Transformations
        if self.transform:
            sample = self.transform(sample)
        else:
            toTensor = ToTensor()
            feature, label = toTensor(sample)
            sample = feature, label

        print(label)
        return sample


class ToTensor(object):
    """Return the output in tensor format.
    """

    def __call__(self, sample):
        feature, label = torch.tensor(sample['feature']), sample['label']
        return feature, label


class Feature_Cube(object):
    """Return a feature cube of desired size.
    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape, augmentation=True):
        self.augmentation = augmentation
        self.cube_shape = cube_shape
        self.num_utterances = cube_shape[0]
        self.num_frames = cube_shape[1]
        self.num_coefficient = cube_shape[2]

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        if self.augmentation:
            # Get some random starting point for creation of the future cube of size (num_frames x num_coefficient x num_utterances)
            # Since we are doing random indexing, the data augmentation is done as well because in each iteration it returns another indexing!
            idx = np.random.randint(feature.shape[0] - self.num_frames, size=self.num_utterances)

        else:
            idx = range(self.num_utterances)

        # Feature cube.
        feature_cube = np.zeros((self.num_utterances, self.num_frames, self.num_coefficient), dtype=np.float32)
        for num, index in enumerate(idx):
            feature_cube[num, :, :] = feature[index:index + self.num_frames, :]

        # return {'feature': feature_cube, 'label': label}
        return {'feature': feature_cube[None, :, :, :], 'label': label}


class CMVN(object):
    """Cepstral mean variance normalization."""

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        feature = speechpy.processing.cmvn(feature, variance_normalization=True)
        return {'feature': feature, 'label': label}


if __name__ == '__main__':
    # dirs = Utils.CopyDataFiles(n_samples=15000)

    cube = Feature_Cube(cube_shape=(20, 80, 40), augmentation=True)

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    db = AudioDataset(c.DATA_ORIGIN + 'train_paths.txt', c.DATA_ORIGIN + 'wav/', transform=transform)

    # trainloader = data.DataLoader(db, batch_size=64)

    N = len(np.genfromtxt(c.DATA_ORIGIN + 'train_paths.txt', dtype='str'))
    print(N)
    dataset = torch.cat([db.__getitem__(idx)[0] for idx in range(N)])
    labels = [db.__getitem__(idx)[1] for idx in range(N)]

    torch.save(dataset, 'dataset.pt')
    torch.save(labels, 'labels.pt')

    labels = np.load('labels.pt')
    labels, freq_occ = np.unique(labels, return_counts=True)

    for label in labels:
        print(label)


    # print(batch_features.shape)
    # print(batch_labels)