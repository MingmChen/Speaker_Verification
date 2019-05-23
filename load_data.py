import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *
from tqdm import tqdm
import speech_feature_extraction.speechpy as speech


# https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/code/0-input/input_feature.py
class AudioDataset(data.Dataset):
    def __init__(self, files_path, audio_dir, indexed_labels, transform=None):
        """
        :param files_path: Path to the .txt file which contains all the file_list
        :param audio_dir:  Directory with all the audio files.
        :param transform:  Optional transform to be applied
                on a sample.
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.indexed = indexed_labels

        # Open the .txt file and create a list from each line.
        content = np.genfromtxt(files_path, dtype='str')
        N = len(content)
        list_files = []
        for x in content:
            sound_file_path = os.path.join(self.audio_dir, x)
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
        if c.NUM_FILES == 0:
            self.sound_files = list_files[:N]
        else:
            self.sound_files = list_files[:c.NUM_FILES]

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        sound_file_path = os.path.join(self.audio_dir, self.sound_files[idx])

        signal = load_wav(sound_file_path)

        # frames = speech.processing.stack_frames(signal,
        #                                         sampling_frequency=c.SAMPLE_RATE,
        #                                         frame_length=0.025,
        #                                         frame_stride=0.01,
        #                                         zero_padding=True)
        #
        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        # power_spectrum = speech.processing.power_spectrum(frames, fft_points=2 * c.NUM_COEF)[:, 1:]

        logenergy = speech.feature.lmfe(signal,
                                        sampling_frequency=c.SAMPLE_RATE,
                                        frame_length=c.FRAME_LEN,
                                        frame_stride=c.FRAME_STEP,
                                        num_filters=c.NUM_COEF,
                                        fft_length=c.NUM_FFT
                                        )

        # Label extraction
        label = self.indexed[self.sound_files[idx][0:7]]
        sample = {
            'feature': logenergy,
            'label': label
        }

        # Apply Transformations
        if self.transform:
            sample = self.transform(sample)
        else:
            toTensor = ToTensor()
            feature, label = toTensor(sample)
            sample = feature, label

        return sample


if __name__ == '__main__':

    dirs = CopyDataFiles(n_samples=5)

    indexed_labels = np.load('labeled_indices.npy').item()

    cube = FeatureCube((80, 40, 20))

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    dataset = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP, indexed_labels, transform=transform)

    N = len(np.genfromtxt(c.DATA_TEMP + 'samples_paths.txt', dtype='str'))

    content = np.genfromtxt(c.DATA_TEMP + 'samples_paths.txt', dtype='str')

    # dataset = [db.__getitem__(idx)[0] for idx in range(N)]

    labels = [dataset.__getitem__(idx)[1] for idx in tqdm(range(N))]

    print(len(np.unique(labels)))
