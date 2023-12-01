# 🌟 DPoser: Diffusion Model as Robust 3D Human Pose Prior 🌟

### [🔗 Project Page](https://dposer.github.io/) | [🎥 Video](https://youtu.be/tbi8nwTaV3M) | [📄 Paper]
#### Authors
[Junzhe Lu](https://scholar.google.com/citations?user=hnJ4NIYAAAAJ), [Jing Lin](https://jinglin7.github.io), [Hongkun Dou](https://scholar.google.com/citations?user=pSNEkEwAAAAJ), [Yulun Zhang](https://yulunzhang.com/), [Yue Deng](https://shi.buaa.edu.cn/yuedeng/en/index.htm), [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)  

<p align="center">
<img src="assets/overview.png" width="1000">
<br>
<em>📊 An overview of DPoser’s versatility and performance across multiple pose-related tasks</em>
</p>

## 📘 1. Introduction  

Welcome to the official implementation of *DPoser: Diffusion Model as Robust 3D Human Pose Prior.* 🚀  
In this repository, we're excited to introduce DPoser, a robust 3D human pose prior leveraging diffusion models. DPoser is designed to enhance various pose-centric applications like human mesh recovery, pose completion, and motion denoising. Let's dive in!

## 🛠️ 2. Setup Your Environment 

- **Tested Configuration**: Our code works great on PyTorch 1.12.1 with CUDA 11.3.

- **Installation Recommendation**:
  ```shell
  conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
  ```
- **Required Python Packages**:
  ```shell
  pip install requirements.txt
  ```
- **Body Models**:
We use the [SMPLX](https://smpl-x.is.tue.mpg.de/) body model in our experiments. Make sure to set the `--bodymodel-path` parameter correctly in scripts like `./run/demo.py` and `./run/train.py` based on your body model's download location.

## 🚀 3. Quick Demo

* **Pre-trained Model**: Grab the pre-trained DPoser model from [here](https://drive.google.com/drive/folders/1hZTF9-WNCz8Wie3LRVdP_3sXTGZle5gN?usp=sharing) and place it in `./pretrained_models`.

* **Sample Data**: Check out `./examples` for some sample files, including 500 body poses from the AMASS dataset and a motion sequence fragment.

* **Explore DPoser Tasks**:

### 🎭 Pose Generation
Generate poses and save rendered images:
  ```shell
  python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task generation
  ```
For videos of the generation process:
  ```shell
  python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task generation_process
  ```

### 🧩 Pose Completion
Complete poses and view results:
  ```shell
  python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task completion --hypo 10 --part right_arm --view right_half
  ```
Explore other solvers like [ScoreSDE](https://github.com/yang-song/score_sde_pytorch) for our DPoser prior:
  ```shell
  python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task completion2 --hypo 10 --part right_arm --view right_half
  ```

### 🌪️ Motion Denoising
Summarize visual results in a video:
  ```shell
  python -m run.motion_denoising --config configs/subvp/amass_scorefc_continuous.py --file-path ./examples/Gestures_3_poses_batch005.npz --noise-std 0.04
  ```

### 🕺 Human Mesh Recovery
Use the detected 2D keypoints from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and save fitting results:
  ```shell
  python -m run.demo_fit --img=./examples/image_00077.jpg --openpose=./examples/image_00077_keypoints.json
  ```


## 🧑‍🔬 4. Train DPoser Yourself
### Dataset Preparation
To train DPoser, we use the [AMASS](https://amass.is.tue.mpg.de/) dataset. You have two options for dataset preparation:

- **Option 1: Process the Dataset Yourself**  
  Download the AMASS dataset and process it using the following script:
  ```shell
  python -m lib/data/script.py
  ```
  Ensure you follow this directory structure:
  ```
  ${ROOT}  
  |-- data  
  |  |-- AMASS  
  |     |-- amass_processed  
  |        |-- version1  
  |           |-- test  
  |              |-- betas.pt  
  |              |-- pose_body.pt  
  |              |-- root_orient.pt  
  |           |-- train  
  |           |-- valid  
  ```

- **Option 2: Use Preprocessed Data**  
  Alternatively, download the processed data directly from [Google Drive](https://drive.google.com/file/d/1TQi_wKxJU3TTcVko-oPlWvp8L12lNR7F/view?usp=sharing).

### 🏋️‍♂️ Start Training
After setting up your dataset, begin training DPoser:
  ```shell
  python -m run.train --config configs/subvp/amass_scorefc_continuous.py --name reproduce
  ```
This command will start the training process. The checkpoints, TensorBoard logs, and validation visualization results will be stored under `./output/amass_amass`.

## 🧪 5. Test DPoser

### Pose Generation
Quantitatively evaluate 500 generated samples using this script:
  ```shell
  python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task generation --metrics
  ```
This will use the [SMPL](https://smpl.is.tue.mpg.de/) body model to evaluate APD and SI following [Pose-NDF](https://github.com/garvita-tiwari/PoseNDF).

### Pose Completion
For testing on the AMASS dataset (make sure you've completed the dataset preparation in Step 4):
  ```shell
  python -m run.completion --config configs/subvp/amass_scorefc_continuous.py --gpus 1 --hypo 10 --sample 10 --part legs
  ```

### Motion Denoising
To evaluate motion denoising on the AMASS dataset, use the following steps:

- Split the `HumanEva` part of the AMASS dataset into fragments using this script:
  ```shell
  python lib/dataset/HumanEva.py --input-dir path_to_HumanEva --output-dir ./data/HumanEva_60frame  --seq-len 60
  ```
- Then, run this script to evaluate the motion denoising task on all sub-sequences in the `data-dir`:
  ```shell
  python -m run.motion_denoising --config configs/subvp/amass_scorefc_continuous.py --data-dir ./data/HumanEva_60frame --noise-std 0.04
  ```

### Human Mesh Recovery
To test on the EHF dataset, follow these steps:

- First, download the EHF dataset from [SMPLX](https://smpl-x.is.tue.mpg.de/).
- Specify the `--data-dir` and run this script:
  ```shell
  python -m run.fitting --config configs/subvp/amass_scorefc_continuous.py --data-dir path_to_EHF --outdir ./fitting_results
  ```

## ❓ Troubleshoots

- `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.`: [Solution here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527)

- `TypeError: startswith first arg must be bytes or a tuple of bytes, not str.`: [Fix here](https://github.com/mcfletch/pyopengl/issues/27). 

## 🙏 Acknowledgement
Big thanks to [ScoreSDE](https://github.com/yang-song/score_sde_pytorch), [GFPose](https://github.com/Embracing/GFPose), and [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE) for their foundational work and code.

## 📚 Reference  

```

```
