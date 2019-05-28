import torch
import torch.nn as nn

class Siamese(nn.Module):
    def __init__(self, LAMBDA, M):
        super(Siamese, self).__init__()
        self.LAMBDA = LAMBDA
        self.M = M

    def forward(self, model, y, output1, output2):
        # y - indicator of samples coming from same speaker
        # output - (n samples, output vector)

        dist = self.l2_dist(output1, output2)
        L_gen = 0.5*dist.pow(2)
        L_imp = 0.5*torch.max(torch.FloatTensor([0.0]).cuda(), self.M-dist).pow(2)

        print(f'Same: {dist[y.byte()].mean().item()}')
        print(f'Not same: {dist[(y-1).abs().byte()].mean().item()}')

        l2_reg = torch.tensor(0.).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)

        loss = 1/y.size()[0]*(y*L_gen+(1-y)*L_imp+self.LAMBDA*l2_reg).sum()

        return loss

    def l2_dist(self, output1, output2):
        return (output1-output2).pow(2).sum(dim=1).sqrt()