# Created by Chen Henry Wu
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.lib.fairface.classifier import Classifier
from model.model_utils import requires_grad


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.classifier = Classifier()
        # Freeze.
        requires_grad(self.classifier, False)

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        # Eval mode for the face classifier
        self.classifier.eval()
        assert split in ['eval', 'test']

        summary = {}
        # Add metrics here.

        # Classify.
        batch_size = 8
        class_type2one_hot = {class_type: [] for class_type in self.args.evaluation.class_types}
        for b in tqdm(range(len(images) // batch_size + 1)):
            start = b * batch_size
            end = (b + 1) * batch_size
            if start == len(images):
                break
            assert start < len(images)
            with torch.no_grad():
                images_b = images[start:end]
                class_type2logits_b = self.classifier(
                    torch.stack(images_b, dim=0).to(self.classifier.device)
                )
                for class_type in self.args.evaluation.class_types:

                    class_type2one_hot[class_type].append(
                        F.one_hot(
                            class_type2logits_b[class_type].argmax(dim=1),
                            num_classes=self.classifier.class_type2num[class_type],
                        ).float().cpu()
                    )
        for class_type in self.args.evaluation.class_types:
            mu = torch.cat(class_type2one_hot[class_type], dim=0).mean(0)
            mu_target = 1 / self.classifier.class_type2num[class_type]
            for idx, mu_i in enumerate(mu):
                label = getattr(self.classifier, f'idx2{class_type}')[idx]
                summary[f'{class_type}_mu_{label}'] = mu_i.item()
            summary[f'{class_type}_kl'] = torch.sum(mu * torch.log(mu / mu_target)).item()
        summary['neg_sum_kl'] = - sum(
            summary[f'{class_type}_kl']
            for class_type in self.args.evaluation.class_types
        )

        return summary

