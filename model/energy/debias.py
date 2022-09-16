# Created by Chen Henry Wu
from collections import OrderedDict

import torch
import torch.nn as nn
from ..lib.fairface.classifier import Classifier

from ..model_utils import requires_grad


class DebiasEnergy(nn.Module):
    def __init__(self, debias, debias_ebm_ckpt):
        super(DebiasEnergy, self).__init__()

        self.debias = debias

        # Set up face classifier
        self.classifier = Classifier()
        # Freeze.
        requires_grad(self.classifier, False)

        # Set up lambda
        self.class_type2lambda = nn.ParameterDict()
        state_dict = torch.load(f"./ckpts/{debias_ebm_ckpt}", map_location='cpu')
        # print(state_dict.keys())
        filtered_state_dict = OrderedDict()
        for class_type in debias:
            self.class_type2lambda[class_type] = nn.Parameter(
                torch.FloatTensor(self.classifier.class_type2num[class_type]).fill_(0)
            )
            filtered_state_dict[class_type] = state_dict[f'module.class_type2lambda.{class_type}']
        self.class_type2lambda.load_state_dict(filtered_state_dict, strict=True)

    @ staticmethod
    def prepare_inputs(img, z, y=None, alpha=None):
        return {
            'img': img,
        }

    def forward(self, img):
        # Eval mode for the image classifier.
        self.classifier.eval()

        # Classify.
        class_type2logits = self.classifier(img)

        # Losses
        debias_loss = torch.zeros(img.shape[0], device=img.device)
        for class_type in self.debias:
            phi = torch.softmax(class_type2logits[class_type], dim=1)
            log_omiga = torch.einsum(
                'c,bc->b',
                self.class_type2lambda[class_type],
                phi,
            )

            debias_loss += (- log_omiga)

        return debias_loss

