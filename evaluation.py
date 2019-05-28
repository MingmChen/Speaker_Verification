from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from model import *
import os
import matplotlib.pyplot as plt


def get_and_plot_k_eer_auc(label, scores, k=1):
    step = int(label.shape[0] / float(k))
    EER_VECTOR = np.zeros((k, 1))
    AUC_VECTOR = np.zeros((k, 1))

    fig = plt.figure()
    ax = fig.gca()

    for split_num in range(k):
        index_start = split_num * step
        index_end = (split_num + 1) * step

        EER_temp, AUC_temp, fpr, tpr = get_eer_auc(label[index_start:index_end], scores[index_start:index_end])

        EER_VECTOR[split_num] = EER_temp
        AUC_VECTOR[split_num] = AUC_temp

        plt.setp(plt.plot(fpr, tpr, label='{} split'.format(split_num)), linewidth=2)

    print("EER=", np.mean(EER_VECTOR) * 100)
    print("AUC=", np.mean(AUC_VECTOR) * 100)

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.title('ROC with {}-fold cross validation'.format(k))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.savefig('eer_auc.png')

    # plt.show()


def get_eer_auc(label, distance):
    fpr, tpr, thresholds = roc_curve(label, distance, pos_label=1)
    auc = roc_auc_score(label, distance)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer, auc, fpr, tpr


class Evaluation:
    def __init__(self, background_model, speaker_models_path):

        self.model = background_model
        self.speaker_models = {}
        for file in os.listdir(speaker_models_path):
            if torch.cuda.is_available():
                self.speaker_models[file.replace('.pt', '')] = (torch.load(speaker_models_path + '/' + file))
            else:
                self.speaker_models[file.replace('.pt', '')] = (torch.load(speaker_models_path + '/' + file,
                                                                           map_location=lambda storage, loc: storage))

    def compute_Similarity(self, utterance, type='cosine_similarity'):
        self.model.eval()
        speaker_features = self.model.create_Speaker_Model(utterance).detach().numpy()

        if type == 'cosine_similarity':

            similarity_vec = np.zeros(len(self.speaker_models))
            assigned_speaker_vec = np.zeros(len(self.speaker_models))

            for index, (key, speaker_model) in enumerate(self.speaker_models.items()):
                similarity_vec[index] = cosine_similarity(speaker_features, speaker_model.detach().numpy())

            assigned_speaker_vec[np.argmax(similarity_vec)] = 1

            # print('the speaker was closer to {}'.format(
            #     list(self.speaker_models.items())[np.argmax(assigned_speaker_vec)][0]))

            return similarity_vec, assigned_speaker_vec


def create_dataset(indexed_labels, origin_file_path):
    from load_data import AudioDataset

    cube_shape = (80, 40, 20)
    cube = FeatureCube(cube_shape)
    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    dataset = AudioDataset(
        origin_file_path,
        c.DATA_ORIGIN,
        indexed_labels=indexed_labels,
        transform=transform)

    return dataset


def evaluate():
    model_path = '/Users/leonidas/Downloads/model_14_percent_best_so_far.pt'

    if not torch.cuda.is_available():
        model = C3D2(100, 1).load_checkpoint(torch.load(model_path, map_location=lambda storage,loc: storage))
    else:
        model = C3D2(100, 1).load_checkpoint(torch.load(model_path))

    dir_path = os.path.join(c.ROOT, 'speaker_models')
    test_set = os.path.join(c.ROOT, '50_first_ids.txt')
    indexed_labels = np.load(c.ROOT + '/50_first_ids.npy', allow_pickle=True).item()

    dataset = create_dataset(indexed_labels=indexed_labels, origin_file_path=test_set)

    eval = Evaluation(model, dir_path)

    speaker_model_ids = list(eval.speaker_models.keys())
    labels = []
    scores = []

    for i in range(len(dataset)):
        features = dataset.__getitem__(i)[0]
        [a, b, cc, d] = features.shape
        s = torch.from_numpy(features.reshape((1, a, b, cc, d)))

        similarity_vec, _ = eval.compute_Similarity(s)
        scores.append(similarity_vec)

        current_id = dataset.sound_files[i][0:7]

        print('correct speaker {} , the speaker was closer to {}'.format(current_id,
                                                                         speaker_model_ids[np.argmax(similarity_vec)]))

        true_label = np.zeros_like(similarity_vec)
        true_label[np.argwhere(current_id in speaker_model_ids)] = 1
        labels.append(true_label)


    labels = np.array(labels)
    scores = np.array(scores)

    # labels = np.array([[1., 0.], [1., 0.]])
    # scores = np.array([[0.7, 0.3], [0.2, 0.8]])
    # print(labels.flatten())

    # fpr, tpr, thresholds = roc_curve(labels[0], scores[0], pos_label=1)
    get_and_plot_k_eer_auc(labels.flatten(), scores.flatten(), k=1)


if __name__ == '__main__':
    evaluate()
