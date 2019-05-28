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
    plt.show()


def get_eer_auc(label, distance):
    fpr, tpr, thresholds = roc_curve(label, distance, pos_label=1)
    auc = roc_auc_score(label, distance)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer2 = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]

    if np.array_equal(eer, eer2):
        print("SAME")
    return eer, auc, fpr, tpr


class Evaluation:
    def __init__(self, background_model, utterance, speaker_models_path):

        self.model = background_model
        self.utterance = utterance
        self.speaker_models = {}
        for file in os.listdir(speaker_models_path):
            if torch.cuda.is_available():
                self.speaker_models[file] = (torch.load(speaker_models_path + '/' + file))
            else:
                self.speaker_models[file] = (torch.load(speaker_models_path + '/' + file,
                                                        map_location=lambda storage, loc: storage))

    def compute_Similarity(self, type='cosine_similarity'):
        self.model.eval()
        self.speaker_features = self.model.create_Speaker_Model(self.utterance).detach().numpy()
        if type == 'cosine_similarity':
            similarity_vec = np.zeros((1, len(self.speaker_models)))
            assigned_speaker_vec = np.zeros((1, len(self.speaker_models)))
            for index, (key, speaker_model) in enumerate(self.speaker_models.items()):
                similarity_vec[0, index] = cosine_similarity(self.speaker_features, speaker_model.detach().numpy())
            assigned_speaker_vec[0, np.where(similarity_vec == np.max(similarity_vec))[1]] = 1
            print('the speaker was closer to {}'.format(
                list(self.speaker_models.items())[np.argmax(assigned_speaker_vec)][0]))
            return similarity_vec, assigned_speaker_vec


if __name__ == '__main__':
    # model = C3D2(n_labels=100, num_channels=1)
    #
    # x = torch.rand((1, 1, 20, 80, 40))
    #
    # dir_path = "C:/Users/anala/Desktop/coursecontent/Speech_and_Speaker/Project/speaker_models"
    # eval = Evaluation(model, x, dir_path)
    # eval.compute_Similarity()
    labels = np.array([[1., 0.], [1., 0.]])
    scores = np.array([[0.7, 0.3], [0.2, 0.8]])
    print(labels.flatten())
    # fpr, tpr, thresholds = roc_curve(labels[0], scores[0], pos_label=1)
    get_and_plot_k_eer_auc(labels.flatten(), scores.flatten(), k=1)
    