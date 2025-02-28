# Generalization Guarantees for Multi-View Representation Learning and Application to Regularization via Gaussian Product Mixture Prior

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/PiotrKrasnowski/Gaussian_Product_Mixture_Priors_for_Multiview_Representation_Learning/blob/main/LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.12.1-%237732a8)

This repository contains the code for our paper [Generalization Guarantees for Multi-View Representation Learning and Application to Regularization via Gaussian Product Mixture Prior]().

## Installation

Most of the requirements are standard Python packages and the CUDA toolkit. A full list of packages can be found below. 

### Requirements:
- Python >= 3.10
- PyTorch >= 1.12.0 (+ CUDA 11.0)
- torchvision >= 0.14 (+ CUDA 11.0)
- matplotlib
- time
- os
- numpy
- numbers

## Reproduction of the results

To reproduce the results reported in our paper, run firstly the script ['main.py']. The full training time may take several hours, but it could be shortened by reducing the number of repetitions from 5 to 1. The training results and the trained models are stored in the directory ['results'], in a folder with a most recent timestamp. 

## Citation

We welcome you to contact us (e-mail: ```p.g.krasnowski@gmail.com```) if you have any problem when reading the paper or reproducing the code.

## Acknowledgement
This code is based on the Variation Information Bottleneck (VIB) implementation of [https://github.com/1Konny/VIB-pytorch.git](https://github.com/1Konny/VIB-pytorch.git). The code, however, has been modified by us substantially.
