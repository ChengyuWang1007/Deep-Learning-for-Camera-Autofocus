# Deep Learning for Camera Autofocus
This repository contains code to build the networks in "Deep Learning for Camera Autofocus" by Chengyu Wang, Qian Huang, Ming Cheng, Zhan Ma and David J. Brady.

(Last updated on: Dec 11, 2020)

* model_calibration.m 

    Matlab code to calibrate the model given two images.

* estimator_discriminator.py

    Keras code to build the estimator and the discriminator in the AF pipeline.
    
    The images used in training can be found:
    
    CLIC dataset: http://www.compression.cc/2018/challenge/
    
    DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

* rl-agent

    Implementation of the reinforcement agent
    
    
