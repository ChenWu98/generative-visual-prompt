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

## Pre-trained checkpoints

### Pre-trained generative models
We provide a unified interface for various pre-trained generative models. Checkpoints for generative models used in this paper are provided below. 
1. StyleGAN2
```shell
cd ckpts/
wget https://www.dropbox.com/s/iy0dkqnkx7uh2aq/ffhq.pt
wget https://www.dropbox.com/s/lmjdijm8cfmu8h1/metfaces.pt
wget https://www.dropbox.com/s/z1vts069w683py5/afhqcat.pt
wget https://www.dropbox.com/s/a0hvdun57nvafab/stylegan2-church-config-f.pt
wget https://www.dropbox.com/s/x1d19u8zd6yegx9/stylegan2-car-config-f.pt
wget https://www.dropbox.com/s/hli2x42ekdaz2br/landscape.pt
```
2. Diffusion Autoencoder
```shell
cd ckpts/
wget https://www.dropbox.com/s/ej0jj8g7crvtb5e/diffae_ffhq256.ckpt
wget https://www.dropbox.com/s/w5y89y57r9nd1jt/diffae_ffhq256_latent.pkl
wget https://www.dropbox.com/s/rsbpxaswnfzsyl1/diffae_ffhq128.ckpt
wget https://www.dropbox.com/s/v1dvsj6oklpz652/diffae_ffhq128_latent.pkl
```
3. StyleNeRF
```shell
cd ckpts/
wget https://www.dropbox.com/s/n80cr7isveh5yfu/StyleNeRF_ffhq_1024.pkl
```
3. BigGAN
```text
# BigGAN will be downloaded automatically
```

### Pre-trained off-the-shelf models
PromptGen allows us to use arbitrary off-the-shelf models to control pre-trained generative models. The off-the-shelf models used in this paper are provided below. 
1. CLIP
```text
# CLIP will be downloaded automatically
```
2. ArcFace IR-SE 50 model, provided by the Colab demo in [this repo](https://github.com/orpatashnik/StyleCLIP)
```shell
cd ckpts/
wget https://www.dropbox.com/s/qg7co4azsv5sacm/model_ir_se50.pth
```
3. DECA model, provided by [this repo](https://github.com/YadiraF/DECA). 
You should first download the [FLAME model](https://flame.is.tue.mpg.de/download.php) (registration is required), 
choose **FLAME 2020** and unzip it, 
copy `generic_model.pkl` into `model/lib/decalib/data/`, and then run the following command 
```shell
wget https://www.dropbox.com/s/972j1vgfd19b6gx/deca_model.tar -O model/lib/decalib/data/deca_model.tar
```
4. FairFace classifier, provided by [this repo](https://github.com/dchen236/FairFace) 
```shell
cd ckpts/
wget https://www.dropbox.com/s/v1rp0uubk30esdh/res34_fair_align_multi_7_20190809.pt
```
5. CelebA classifier, trained by ourselves
```shell
cd ckpts/
wget https://www.dropbox.com/s/yzc8ydaa4ggj1zs/celeba.zip
unzip celeba.zip 
```

### Pre-trained beta-hat models (for the moment constraint)
For the moment constraint experiments, one need to train the beta-hat model based on the given pre-trained generative model and moment constraint. Although you can train the beta-hat models following our instruction described later, we provide the pre-trained checkpoints here.
1. FFHQ (1024) beta-hat model
```shell
cd ckpts/
wget https://www.dropbox.com/s/htdfv5w1xzsnajj/ffhq_debias.bin
```
2. MetFaces (1024) beta-hat model
```shell
cd ckpts/
wget https://www.dropbox.com/s/j2z9lha15mb2hfj/metfaces_debias.bin
```

## Contact
[Issues](https://github.com/ChenWu98/Generative-Visual-Prompt/issues) are welcome if you have any question about the code. 
If you would like to discuss the method, please contact [Chen Henry Wu](https://github.com/ChenWu98).

<a href="https://github.com/ChenWu98"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a>
