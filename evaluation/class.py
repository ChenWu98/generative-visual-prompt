# Created by Chen Henry Wu
import torch
import torch.nn as nn
from tqdm import tqdm
from model.lib.celeba.classifier import Classifier
from collections import Counter


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.classifiers = nn.ModuleList()
        for class_ in meta_args.ClassEnergy.classes:
            self.classifiers.append(Classifier(class_))

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images (or None), or list of tuples of images (or tuples of None)
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']
        self.classifiers.eval()
        self.classifiers.to(model.device)

        n_samples = 1024
        batch_size = 16
        N = n_samples // batch_size
        if n_samples % batch_size != 0:
            N += 1
        results = []
        for batch in tqdm(range(N)):
            with torch.no_grad():
                generated_images = model.forward_with_size(batch_size)
                preds = []
                for classifier in self.classifiers:
                    logits = classifier(generated_images)
                    preds.append((logits > 0.5).long().cpu())
                preds = torch.stack(preds, dim=1)
                results.append(preds)

        summary = Counter([str(tuple(x.tolist())) for x in torch.cat(results, dim=0)])

        return summary

