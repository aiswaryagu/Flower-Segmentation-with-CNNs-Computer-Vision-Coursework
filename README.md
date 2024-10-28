# Flower Segmentation with CNNs - Computer Vision Coursework
This project focuses on image segmentation using Convolutional Neural Networks (CNNs) applied to the Oxford Flower Dataset. The objective is to classify and segment flowers from background elements within images. The tasks are to use an existing segmentation model and build a custom CNN model to perform segmentation on RGB images sized 256x256x3.

## Objectives
Model Reuse: Adapt an existing segmentation model to work with the dataset, performing necessary re-training.

Custom CNN Development: Build a CNN from scratch and train it specifically for this segmentation task.

Files Included:

segmentationExist.m: MATLAB code for the pre-trained model adaptation.

segmentexistnet: Pre-trained model file for the reused network.

segmentationOwn.m: MATLAB code for the custom CNN architecture.

segmentownnet: Model file for the custom CNN network.

Note: The file segmentexist.mat is not included here due to size constraints.

### Dataset
The Oxford Flower Dataset (17 classes) is used, but only a subset of images with segmentation labels is leveraged for training. Classes are simplified to "flower" and "background" for segmentation purposes.
