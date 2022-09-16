# Created by Chen Henry Wu
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Face classifier preprocess.
        self.classifier_process = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Setup classifier.
        self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
        self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
        self.model_fair_7.load_state_dict(torch.load('ckpts/res34_fair_align_multi_7_20190809.pt'))

        self.class_types = ['race', 'gender', 'age']
        self.class_type2num = {
            'race': 7,
            'gender': 2,
            'age': 9,
        }

    def forward(self, img):
        # Classifier process.
        img = self.classifier_process(img)

        outputs = self.model_fair_7(img)

        class_type2logits = {
            'race': outputs[:, :7],
            'gender': outputs[:, 7: 9],
            'age': outputs[:, 9: 18],
        }

        return class_type2logits

    @property
    def idx2race(self):
        return ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

    @property
    def idx2gender(self):
        return ['Male', 'Female']

    @property
    def idx2age(self):
        return ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    @property
    def device(self):
        return next(self.parameters()).device
