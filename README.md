# A deep learning–based method for the design of microstructural materials

This is my PhD research project on applying deep learning method, specifically Neural Networks to perform microstructural inverse design. The corresponding research paper has been published in the Journal of Structural and Multidisciplinary Optimization. The corresponding journal paper can be found at the following link: https://link.springer.com/article/10.1007/s00158-019-02424-2

The architecture and the implementation of the deep learning models are explained in the text of journal paper, with the corresponding codes for the neural networks provided in this repository.


## Usage
As explained in the paper, the inverse design model is made up of two different neural networks:
(i) Design candidate generator
(ii) Design evaluator
These neural networks are written in Python by using the library of Tensorflow, particularly Tensorflow 1. The results from the journal paper can be partially recreated (testing cases) by using the testing data and also the pretrained checkpoint for the models as supplied below. The models constructed can be used for design applications with different geometrical constraint as well as properties of interest, provided that corresponding training data can be prepared/available.

### Design Candidate Generator
The Deep Convolutional Generative Adversarial Network can be trained by providing the data for training and running main.py in the DCGAN folder. To generate new images based on the pretrained checkpoint, make sure that the checkpoint folder is located in the same directory. The model.py file should also be modified accordingly.

### Design Evaluator
The Convolutional Neural Network for Compliance Tensor Prediction can be demonstrated on the test dataset by running the cnn_compliance.py file. For different dataset (eg: Ellipse or Circle & Square), the code in the file should be modified according to comment in the code.

### Testing Data and Checkpoint File

The test dataset for the model can be downloaded from the link: https://1drv.ms/f/s!Ai_nIBFtgsTMgTrizTMZYov-7VD2
The checkpoint file for the pretrained CNN and DCGAN can be downloaded from the link: https://1drv.ms/f/s!Ai_nIBFtgsTMa3haiYUcgMtCkzI
The test datasets are suitable to used for: (i) Convolutional Neural Network for Compliance Prediction (ii) Inverse Design Model
