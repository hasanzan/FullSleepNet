# Multi-task learning for sleep arousal and stage detection using fully convolutional networks: FullSleepNet
## (This repository will be updated when the paper is accepted)
## Introduction
Source code for FullSleepNet developed for the paper "Multi-task learning for sleep arousal and stage detection using fully convolutional networks: FullSleepNet".

## Description
The model was trained and evaluated on two datasets: Sleep Heart Health Study (SHHS) and Multi-Ethnic Study of Atherosclerosis (MESA)

The architecture of FullSleepNet:
![image](https://user-images.githubusercontent.com/129799320/229633134-35311773-1f99-4b75-bf3f-29f7f9037ab9.png)

## Training
Model can be retrained using two Jupyter notebooks: `main_shhs.ipynb` and `main_mesa.ipynb`. Note that PSG data and labels should be downloaded from https://sleepdata.org/datasets/shhs/ and https://sleepdata.org/datasets/mesa. Files should be placed in `/shhs/data/`, `/shhs/labels/` and `/mesa/data/`, `/mesa/labels/`. 

## Evaluation
Model can be retrained using two Jupyter notebooks: `main_shhs.ipynb` and `main_mesa.ipynb`. Weights for full model trained as explained in the paper can be found in https://github.com/hasanzan/FullSleepNet/edit/main/weights.

## Models
Models used in the paper can be found in https://github.com/hasanzan/FullSleepNet/tree/main/models.

## Cite as
It will be added
