# Adversarial Parametric Pose Prior

This is a Pytorch implementation of the CVPR'22 paper "Adversarial Parametric Pose Prior". 

You can find the paper [here](https://arxiv.org/abs/2112.04203).

<p align="center"> <img src="demos/demo.gif" alt="animated" width=80%/></p>

<h4 align="center">Examples of spherical interpolations when sampling from GAN-S pose prior</h4>

_________

## Installation

We provide the conda environment for Linux. To create and activate it, do:

```
conda env create -f environment.yaml
conda activate adv_prior
```

The main requirements are:
* python=3.8, numpy, matplotlib
* scipy, scikit-learn, jupyter
* easydict, pyyaml, tqdm
* pytorch=1.7, cudatoolkit=10.2
* pytorch3d
* smplx
* torchgeometry

As for SMPL mesh, put the `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` pickle file from [SMPLIFY_CODE_V2.ZIP](https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip) in `data/`. Rename the file, replacing `basicModel` with `smpl`.

## Technical Details

**Note**: We make use of the torchgeometry library which contains an error. For more details please see [this StackOverflow thread](https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported). Essentially, the file `{ANACONDA_HOME}/envs/adv_prior/lib/python3.8/site-packages/torchgeometry/core/conversions.py` must be updated
in the function `rotation_matrix_to_quaternion`:
- line 302: `mask_c1 = mask_d2 * ~(mask_d0_d1)` ~~`mask_d2 * (1 - mask_d0_d1)`~~ 
- line 303: `mask_c2 = ~(mask_d2) * mask_d0_nd1` ~~`(1 - mask_d2) * mask_d0_nd1`~~ 
- line 304: `mask_c3 = (~(mask_d2)) * (~(mask_d0_nd1))` ~~`(1 - mask_d2) * (1 - mask_d0_nd1)`~~ 

_________

## Training

To launch the training of the GAN-S model on AMASS data, run: 
```
python run/main.py --cfg experiments/train_gan.yaml
```
_______

## Demos

We provide two short demo jupyter notebooks for sampling from and interpolating in the latent space. 
All details can be found [here](demos/README.md).