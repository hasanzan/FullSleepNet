# Multi-Task Learning for Arousal and Sleep Stage Detection Using Fully Convolutional Networks
## $\text{\textcolor{red}{(This repository will be updated when the paper is accepted)}}$

## Introduction
Source code for FullSleepNet developed for the paper "Multi-task learning for sleep arousal and stage detection using fully convolutional networks".

## Description
The model was trained and evaluated on two datasets: Sleep Heart Health Study (SHHS) and Multi-Ethnic Study of Atherosclerosis (MESA)

The architecture of FullSleepNet:
![image](https://user-images.githubusercontent.com/129799320/230777550-aecd1f3c-aaa0-4dd6-8ed1-ce32ff0f5090.png)

## Training
The model can be trained from scratch using two Jupyter notebooks: `main_shhs.ipynb` and `main_mesa.ipynb`. Note that PSG data and labels should be downloaded from https://sleepdata.org/datasets/shhs/ and https://sleepdata.org/datasets/mesa. Files should be placed in `/shhs/data/`, `/shhs/labels/` and `/mesa/data/`, `/mesa/labels/`. 

## Evaluation
Model can be evaluated using two Jupyter notebooks: `main_shhs.ipynb` and `main_mesa.ipynb` after training or loading weights. Weights of pre-trained models (models with all modules trained on SHHS or MESA) can be found in https://github.com/hasanzan/FullSleepNet/edit/main/weights. Files should be unzipped and loaded as `model.load_weights("shhs_model-C8-L3A")` or `model.load_weights("mesa_model-C8-L3A")`.

## Models
Code for FullSleepNet and its variants used in the paper can be found in https://github.com/hasanzan/FullSleepNet/tree/main/models.

## Examples
![figure3 - Kopya](https://user-images.githubusercontent.com/129799320/230774935-ac0586e2-4b80-41f2-a8a1-487bf914c993.png)
![figure5 - Kopya](https://user-images.githubusercontent.com/129799320/230774948-206439c1-89da-4519-b3b6-664084ddc71d.png)

## Cite as
It will be added.
