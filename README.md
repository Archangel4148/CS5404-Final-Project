# SPAR3D Robustness Evaluation
By Simon Edmunds and Joshua Santy

This repository contains the code used in our evaluation of the robustness of **SPAR3D**, a single-image 3D reconstruction method, when its input is subjected to image distortions. Our goal is to measure how sensitive SPAR3D is to changes in image quality, including blur, noise, and exposure boosting, and to evaluate how these distortions affect reconstruction accuracy.

## Overview

This project includes:
- A fully automated evaluation pipeline for running SPAR3D on distorted images
- Implementations of three image distortion types (Gaussian blur, Gaussian noise, exposure boosting)
- Chamfer Distance and F-score metric computation for reconstructed point clouds
- Scripts for running experiments across multiple objects and distortion levels
- Tools for generating summary tables and plots used in our report
- R Studio code for running statistical analysis on the final results
