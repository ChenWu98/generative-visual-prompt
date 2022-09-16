# Created by Chen Henry Wu
import torch.nn as nn


class PriorYEnergy(nn.Module):
    def __init__(self, y_scale):
        super(PriorYEnergy, self).__init__()

        self.y_scale = y_scale

    @ staticmethod
    def prepare_inputs(img, z, y, alpha=None):
        return {
            'y': y,
        }

    def forward(self, y):
        prior_y_loss = 0.5 * (y ** 2).sum(1) / (self.y_scale ** 2)

        return prior_y_loss
