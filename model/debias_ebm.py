# Created by Chen Henry Wu
import torch
import torch.nn as nn

from .lib.fairface.classifier import Classifier
from .model_utils import requires_grad, MAX_SAMPLE_SIZE
from .gan_wrapper.get_gan_wrapper import get_gan_wrapper


class DebiasEBM(nn.Module):

    def __init__(self, args):
        super(DebiasEBM, self).__init__()

        self.args = args.model

        # Set up gan_wrapper
        self.gan_wrapper = get_gan_wrapper(args.gan)
        # Freeze.
        requires_grad(self.gan_wrapper, False)

        # Set up face classifier
        self.classifier = Classifier()
        # Freeze.
        requires_grad(self.classifier, False)

        # Set up lambda
        self.class_type2lambda = nn.ParameterDict()
        for class_type in args.model.debias:
            self.class_type2lambda[class_type] = nn.Parameter(
                torch.FloatTensor(self.classifier.class_type2num[class_type]).fill_(0)
            )

        # Fixed noise for better visualization.
        self.register_buffer(
            "fixed_z",
            torch.randn(MAX_SAMPLE_SIZE, self.gan_wrapper.latent_dim),
        )

    def get_z_gaussian(self, sample_id=None):
        if self.training:
            bsz = sample_id.shape[0]
            z = torch.randn(bsz, self.gan_wrapper.latent_dim, device=self.device)
        else:
            assert sample_id.dim() == 1
            z = self.fixed_z[sample_id, :]

        return z

    def forward(self, sample_id):

        # Eval mode for the gan_wrapper and the face classifier.
        self.gan_wrapper.eval()
        self.classifier.eval()

        z = self.get_z_gaussian(sample_id=sample_id)  # (B, style_dim)

        with torch.no_grad():
            small_batch = 16
            assert sample_id.shape[0] % small_batch == 0
            class_type2logits = None
            for b in range(sample_id.shape[0] // small_batch):
                z_b = z[b * small_batch:(b + 1) * small_batch, :]

                img_b = self.gan_wrapper(z=z_b)

                # Classify.
                class_type2logits_b = self.classifier(img_b)
                class_type2logits = class_type2logits_b if class_type2logits is None else \
                    {
                        class_type: torch.cat(
                            [
                                class_type2logits[class_type],
                                class_type2logits_b[class_type],
                            ],
                            dim=0,
                        )
                        for class_type in self.classifier.class_types
                    }

        # Losses
        losses = dict()
        weighted_loss = torch.zeros_like(sample_id).float()
        for class_type in self.args.debias:
            phi = torch.softmax(class_type2logits[class_type], dim=1)
            omiga = torch.exp(
                torch.einsum(
                    'c,bc->b',
                    self.class_type2lambda[class_type],
                    phi,
                )
            )
            mu_hat = torch.einsum('bc,b->c', phi, omiga) / omiga.sum(0)
            mu_target = 1 / self.classifier.class_type2num[class_type]

            mu_loss = (mu_hat - mu_target).pow(2).sum(0, keepdim=True)
            # As batch shape.
            mu_loss = mu_loss.expand(sample_id.shape[0]).contiguous()
            losses[f'{class_type}_loss'] = mu_loss
            weighted_loss += mu_loss
            for idx, lambda_i in enumerate(self.class_type2lambda[class_type]):
                label = getattr(self.classifier, f'idx2{class_type}')[idx]
                # As batch shape.
                losses[f'{class_type}_lambda_{label}'] = lambda_i[None].expand(sample_id.shape[0]).contiguous()
                losses[f'{class_type}_loss_{label}'] = (mu_hat - mu_target).pow(2)[idx:idx + 1].expand(sample_id.shape[0]).contiguous()

        return None, weighted_loss, losses

    @property
    def device(self):
        return next(self.parameters()).device


Model = DebiasEBM
