import numpy as np
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from utils import *
from model import *
import os
import matplotlib.pyplot as plt


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
    model = C3D2(n_labels=100, num_channels=1)
    x = torch.rand((1, 1, 20, 80, 40))
    dir_path = "C:/Users/anala/Desktop/coursecontent/Speech_and_Speaker/Project/speaker_models"
    eval = Evaluation(model, x, dir_path)
    eval.compute_Similarity()
