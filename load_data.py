import torchvision.transforms as transforms
import torch.utils.data as data
from Utils import *

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

        signal = load_wav(sound_file_path)

        frames = speechpy.processing.stack_frames(signal,
                                                  sampling_frequency=c.SAMPLE_RATE,
                                                  frame_length=0.025,
                                                  frame_stride=0.01,
                                                  zero_padding=True)

        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * c.NUM_COEF)[:, 1:]

        logenergy = speechpy.feature.lmfe(signal,
                                          sampling_frequency=c.SAMPLE_RATE,
                                          frame_length=c.FRAME_LEN,
                                          frame_stride=c.FRAME_STEP,
                                          num_filters=c.NUM_COEF,
                                          fft_length=c.NUM_FFT
                                          )

        # Label extraction
        # label = int(self.sound_files[idx][2:7])
        label = idx
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
    dirs = CopyDataFiles(n_samples=10)

    cube = FeatureCube((80, 40, 20))

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    db = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP , transform=transform)

    N = len(np.genfromtxt(c.DATA_TEMP + 'samples_paths.txt', dtype='str'))
    # print(N)
    dataset = [db.__getitem__(idx)[0] for idx in range(N)]
    labels = [db.__getitem__(idx)[1] for idx in range(N)]

    data_point = db.__getitem__(0)[0]

    print(data_point.shape)
