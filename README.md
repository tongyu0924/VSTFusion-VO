# Monocular Visual Odometry via Swin-based Multimodal Fusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aofrancani/TSformer-VO/blob/main/LICENSE)
 
This project is a **fork of [aofrancani/TSformer-VO](https://github.com/aofrancani/TSformer-VO)** with custom modifications based on the SWFormer-VO architecture.

---

## This Fork: Swin-based Multimodal Extension

This project integrates ideas from TSformer-VO and SWFormer-VO,
introducing a Swin-based multimodal VO framework with the following modifications:

- Replacing the original Timesformer backbone with **Video Swin Transformer** (stages 1~3).
- Introducing **early fusion of RGB and Depth embeddings** before feature encoding.
- Retaining the temporal structure of the original architecture while improving its adaptability to multimodal inputs.
- Evaluating on **KITTI Odometry**, showing improvements over the original model in three key metrics:  
  - **↓1.9%** Translational Error  
  - **↓11.5%** Absolute Trajectory Error (ATE)  
  - **↓8.1%** Relative Pose Error (RPE)

---

## Abstract (for this extension)

*This work builds upon TSformer-VO by incorporating early-stage fusion of RGB and depth inputs via the Video Swin Transformer. We explore the impact of a hierarchical attention-based backbone in visual odometry tasks, treating the problem as a spatiotemporal sequence understanding challenge. Our model demonstrates improved accuracy on the KITTI dataset, highlighting the benefits of Swin-based feature extraction in multimodal settings.*


<img src="tsformer-vo.jpg" width=1000>

## Contents
1. [Dataset](#1-dataset)
2. [Pre-trained models](#2-pre-trained-models)
3. [Setup](#3-setup)
4. [Usage](#4-usage)
5. [Evaluation](#5-evaluation)


## 1. Dataset
Download the [KITTI odometry dataset (grayscale).](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

In this work, we use the `.jpg` format. You can convert the dataset to `.jpg` format with [png_to_jpg.py.](https://github.com/aofrancani/DPT-VO/blob/main/util/png_to_jpg.py)

Create a simbolic link (Windows) or a softlink (Linux) to the dataset in the `dataset` folder:

- On Windows:
```mklink /D <path_to_your_project>\TSformer-VO\data <path_to_your_downloaded_data>```
- On Linux: 
```ln -s <path_to_your_downloaded_data> <path_to_your_project>/TSformer-VO/data```

The data structure should be as follows:
```
|---TSformer-VO
    |---data
        |---sequences_jpg
            |---00
                |---image_0
                    |---000000.png
                    |---000001.png
                    |---...
                |---image_1
                    |...
                |---image_2
                    |---...
                |---image_3
                    |---...
            |---01
            |---...
		|---poses
			|---00.txt
			|---01.txt
			|---...
```

## 2. Pre-trained models

Here you find the checkpoints of our trained-models.

**Google Drive folder**: [link to checkpoints in GDrive](https://drive.google.com/file/d/1zdUS6J6jkKsSrhkCWJV7k3cJTQIqnUCi/view?usp=sharing)


## 3. Setup
- Create a virtual environment using Anaconda and activate it:
```
conda create -n tsformer-vo python==3.8.0
conda activate tsformer-vo
```
- Install dependencies (with environment activated):
```
pip install -r requirements.txt
```

## 4. Usage

**PS**: So far we are changing the settings and hyperparameters directly in the variables and dictionaries. As further work, we will use pre-set configurations with the `argparse` module to make a user-friendly interface.

### 4.1. Training

In `train.py`:
- Manually set configuration in `args` (python dict);
- Manually set the model hyperparameters in `model_params` (python dict);
- Save and run the code `train.py`.

### 4.2. Inference

In `predict_poses.py`:
- Manually set the variables to read the checkpoint and sequences.

| **Variables**   | **Info**                                                                                                             |
|-----------------|----------------------------------------------------------------------------------------------------------------------|
| checkpoint_path | String with the path to the trained model you want to use for inference.  Ex: checkpoint_path = "checkpoints/Model1" |
| checkpoint_name | String with the name of the desired checkpoint (name of the .pth file).  Ex: checkpoint_name = "checkpoint_model2_exp19" |
| sequences       | List with strings representing the KITTI sequences.  Ex: sequences = ["03", "04", "10"]                              |

### 4.3. Visualize Trajectories
In `plot_results.py`:
- Manually set the variables to the checkpoint and desired sequences, similarly to [Inference](#42-inference)


## 5. Evaluation
The evaluation is done with the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval). Please go to the [evaluation repository](https://github.com/Huangying-Zhan/kitti-odom-eval) to see more details about the evaluation metrics and how to run the toolbox.


## Citation
Please cite our paper you find this research useful in your work:

```bibtex
@article{Francani2023,
  title={Transformer-based model for monocular visual odometry: a video understanding approach},
  author={Fran{\c{c}}ani, Andr{\'e} O and Maximo, Marcos ROA},
  journal={arXiv preprint arXiv:2305.06121},
  year={2023}
}
```

## References

Code adapted from [TimeSformer](https://github.com/facebookresearch/TimeSformer). 

Check out our previous work on monocular visual odometry: [DPT-VO](https://github.com/aofrancani/DPT-VO)

 
