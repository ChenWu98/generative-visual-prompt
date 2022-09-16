# Created by Chen Henry Wu
import torch
from torch import nn

from ..model_utils import requires_grad
from ..lib.id_recognition.model_irse import Backbone


class IDEnergy(nn.Module):
    def __init__(self, ir_se50_weights):
        super(IDEnergy, self).__init__()

        self.id_loss = IDLoss(ir_se50_weights)

        # Freeze.
        requires_grad(self.id_loss, False)

    @ staticmethod
    def prepare_inputs(img, img_0, alpha):
        return {
            'img': img,
            'img_0': img_0,
        }

    def forward(self, img, img_0):
        # Eval mode for the IDLoss module.
        self.id_loss.eval()

        id_loss = self.id_loss(img, img_0)

        return id_loss


class IDLoss(nn.Module):
    """This loss follows StyleCLIP. """
    def __init__(self, ir_se50_weights):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        # Eval mode.
        self.facenet.eval()

        x_feats, y_feats = self.extract_feats(x), self.extract_feats(y)  # Features are already l2 normalized.
        diff_target = torch.einsum('bh,bh->b', x_feats, y_feats)
        loss = 1 - diff_target

        return loss
