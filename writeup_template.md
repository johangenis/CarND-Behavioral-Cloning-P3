# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The NVIDIA [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) Network Architecture was used to solve this problem. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 41). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 17-25, and 52-71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination circuits around both tracks in clock-, and anti-clockwise directions, and kept to the centre of the road as much as possible.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed David's lectures to build a network based on NVIDIA's architecture, as stated above. 

#### 2. Final Model Architecture

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA.  

The NVIDIA Model is a deep convolution network which is well suited for for the problem presented by the project.  The NVIDIA model is well documented, which enabled me to concentrate on trying to fine tune the network and obtaining good data.

The following adjustments were added to the model:

- Lambda layer to normalize input images to avoid saturation and improving gradients work.
- A dropout layer to avoid overfitting.
- ELU as the activation function for every layer, except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output) 

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1   |
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1         |
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 1, 18, 64) |0       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |dropout_1        |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |252219  |                 |



![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used a combination circuits around both tracks in clockwise (once for each track), and anti-clockwise(once for each track) directions, and kept to the centre of the road as much as possible. I found it not neccessary to record recovery scenarios.

After the collection process, I had 72 165 data points. I then preprocessed this data by cropping, resizing and converting the frames from RGB to YUV. (Lines 38 - 45 of utils.py)

Finally, the data was randomly shuffled and 20% of the data was put into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 171 as evidenced by the fact the I kept the best models while the network was training and the model produced by Epoch 171 turned out to be the model that completed both tracks successfully. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Lessons learned

It took me much longer to complete this project than anticipated.  The points below could have saved a lot of time, had I known them before starting the project:

 - Get a game controller to record training data - using the keyboard results in bad data
 - Bad data will never produce a good model, no matter how much it is trained
 - Do not be afraid to traing for more Epochs than what is estimated
 - Train on Amazon's GPU's
 
