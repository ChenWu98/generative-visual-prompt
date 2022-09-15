# Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models

Official PyTorch implementation of our **NeurIPS 2022** paper <br>
**Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models** <br>
Chen Henry Wu, Saman Motamed, Shaunak Srivastava, Fernando De la Torre <br>
Carnegie Mellon University <br>
NeurIPS 2022

---

<img src="docs/first_fig.png" align="middle">


Generative models (e.g., GANs and diffusion models) learn the underlying data distribution in an unsupervised manner. 
However, many applications of interest require sampling from a specific region of the generative model's output space or evenly over a range of characteristics. 
To allow efficient sampling in these scenarios, we propose **Generative Visual Prompt (PromptGen)**, a framework to achieve distributional control over pre-trained generative models by incorporating knowledge of arbitrary off-the-shelf models. 
PromptGen defines control as an energy-based model (EBM) and samples images in a feed-forward manner by approximating the EBM with invertible neural networks, avoiding optimization at inference. 
We demonstrate how PromptGen can control several generative models (e.g., **StyleGAN2**, **diffusion autoencoder**, **StyleNeRF**, **NVAE**) using various off-the-shelf models: 
1. With the CLIP model, PromptGen can sample images guided by the text.
2. With image classifiers, PromptGen can de-bias generative models across a set of attributes.
3. With inverse graphics models, PromptGen can sample images of the same identity in different poses. 
4. Finally, PromptGen reveals that the CLIP model shows "reporting bias" when used as control, and PromptGen can further de-bias this controlled distribution in an iterative manner.

<img src="docs/pipeline_overview.png" align="middle">

PromptGen requires _no training data_, and the only supervision comes from off-the-shelf models that help define the control. 
It samples images in a _feed-forward_ manner, which is highly efficient, and it also _stands alone_ at inference, meaning that we can discard the off-the-shelf models after training.
PromptGen not only offers _generality_ for algorithmic design and _modularity_ for control composition, 
but also enables _iterative_ controls when some controls are contingent on others. 

<img src="docs/overview.png" align="middle">

## Dependencies

1. Create environment by running
```shell
conda env create -f environment.yml
conda activate generative_prompt
pip install git+https://github.com/openai/CLIP.git
```
2. Install `torch` and `torchvision` based on your CUDA version. 
3. Install [PyTorch 3D](https://github.com/facebookresearch/pytorch3d) (only needed for experiments with StyleNeRF).
4. Set up [wandb](https://wandb.ai/) for logging (registration is required). You should modify the ```setup_wandb``` function in ```main.py``` to accomodate your wandb credentials.

## Contact
[Issues](https://github.com/ChenWu98/Generative-Visual-Prompt/issues) are welcome if you have any question about the code. 
If you would like to discuss the method, please contact [Chen Henry Wu](https://github.com/ChenWu98).

<a href="https://github.com/ChenWu98"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a>
