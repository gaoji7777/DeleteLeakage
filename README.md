# Code for Deletion inference and Reconstruction

## Introduction

This repo contains the codes for the paper [Deletion Inference, Reconstruction, and Compliance in Machine (Un)Learning](https://arxiv.org/abs/2202.03460). We present codes for three different experiments in the paper, including 
- Deletion inference on CIFAR-10 and CIFAR-100 (in folder [DeletionInference](https://github.com/gaoji7777/DeleteLeakage/tree/main/DeletionInference))
- Deletion reconstruction for images (in folder [ImageReconstruction](https://github.com/gaoji7777/DeleteLeakage/tree/main/ImageReconstruction)), 
- Deletion reconstruction for language models(in folder [LanguageReconstruction](https://github.com/gaoji7777/DeleteLeakage/tree/main/LanguageReconstruction)).

## Required environment

We use Python 3 for the codes.

- pytorch > 1.0 (and torchvision, which should be installed as part of pytorch).
- tqdm 

## Running the code

Just cd into the experiment folder and follow the instruction of each README. Note that running the code will create a folder called 'data', and may take 700MB space.
