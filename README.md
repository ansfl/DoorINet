# DoorINet
![ANSFL_logo](https://github.com/ansfl/DoorINet/assets/89016122/f98fb44e-5e0e-4cd8-b4cf-c5417be76161)

This repository contains the dataset, source code and pretrained weights for DoorINet architecture introduced in the paper **DoorINet: A Deep Learning Inertial Framework for Door-Mounted IoT Applications**.

## Introduction
The Internet of Things (IoT) is essentially a networkof devices that can collect information and connect to and communicate with each other without human interaction.
In non-navigational applications (such as most IoT applications) inertial sensors readings are used as input to attitude and heading reference system (AHRS) algorithms that provide orientation of a device. That is, the three Euler angles with respect to a fixed coordinate frame, which uniquely determine the orientation.

Here we introduce and address the problem of door mounted IMUs. There, the goal is to determine the opening angle of a door. This information may be utilized in several IoT applications such as smart homes, smart offices or building management. To cope with low magnetometer performance in indoor enthronements, we propose **DoorINet** - a deep learning end-to-end framework for estimating the heading angle of door-mounted IMU using only accelerometers and gyroscopes. Two versions of **DoorINet** were examined: **AG-DoorINet** that
takes accelerometer and gyroscope measurements, and **G-DoorINet** that takes only gyroscope readings.

All the data processing and operations was implemented in **Python** programming language using also **numpy** and **pandas** libraries. **imufusion** library was used for implementing and modifying Madgwick algorithm. Deep learning models architectures and training were implemented using **PyTorch** framework.

_**utils**_ folder in this repository contains all the functions for data processing, _**models**_ foder contains DoorINet models architecture and training functions as well as pretrained models weights, _**dataset**_ folder has train and test dataset.

## Requirements

python >= 3.8

numpy >= 1.26

pandas >= 2.1

pytorch >= 2.0

imufusion >= 1.1.1

## Models architectures
**G-DoorINet** architecture consists of bi-directional GRU layers and fully-connected (dense) layers. It takes a series of 20 consequent gyroscope measurements as an input and outputs a heading angle increment over the input series taken.
![Figure_3_1](https://github.com/ansfl/DoorINet/assets/89016122/ca90a2b7-cc90-4ab3-91a5-ff640724552b)

**AG-DoorINet** architecture also consists of bi-directional GRU layers and fully-connected (dense) layers. It takes a series of 20 consequent gyroscope and accelerometer measurements as an input and outputs a heading angle increment over the input series.
![Figure_3](https://github.com/ansfl/DoorINet/assets/89016122/c536ca50-f072-4c44-bbfa-48a16404d49e)

## Setup 

* Install all the necessary packages (see **Requirements** section). 
* Clone this repository.
* Download models and dataset from [Google drive](https://drive.google.com/drive/folders/11T_5DR6Rnr8eeFteVZ_w3SqR17zyQix1?usp=sharing)
* Use train.py and test.py for training and testing of the models
* Use inference.py for loading the trained model

## Citation

_to be written soon_

