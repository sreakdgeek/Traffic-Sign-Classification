**Traffic Sign Recognition** 

### Introduction

Traffic Sign Classifier is trained on German Traffic signs dataset and classifies/recognizes new traffic images as one of the 43 traffic signs. Below are the steps to build Traffic Sign Classifier pipeline:

*	Exploratory Data Analysis
*	Data Augmentation
*	Data Pre-processing
*	Build Convolutitional Neural Network Architecture
*	Train, validate, test and fine tune model and Network architecture
*	Predict sign for new images and evaluate performance
*	Visualize convolutional neural network to understand what each layer learns

[//]: # (Image References)

[image1]: ./examples/TrafficSign1.jpg "Visualization"
[image2]: ./examples/TrafficSign2.jpg "Grayscaling"
[image3]: ./examples/TrafficSign3.jpg.png "Random Noise"
[image4]: ./examples/TrafficSign4.jpg "Traffic Sign 1"
[image5]: ./examples/TrafficSign5.jpg "Traffic Sign 2"
[image6]: ./examples/TrafficSign6.jpg "Traffic Sign 3"
[image7]: ./examples/TrafficSign7.jpg "Traffic Sign 4"
[image8]: ./examples/TrafficSign8.jpg "Traffic Sign 5"
[image9]: ./examples/TrafficSign9.jpg "Visualization"
[image10]: ./examples/TrafficSign10.jpg "Visualization"
[image11]: ./examples/TrafficSign11.jpg "Visualization"
[image12]: ./examples/TrafficSign12.jpg "Grayscaling"
[image13]: ./examples/TrafficSign13.jpg "Random Noise"
[image14]: ./examples/TrafficSign14.jpg "Traffic Sign 1"
[image15]: ./examples/TrafficSign15.jpg "Traffic Sign 2"
[image16]: ./examples/TrafficSign16.JPG "Traffic Sign 3"
[image17]: ./examples/TrafficSign17.jpg "Traffic Sign 4"
[image18]: ./examples/TrafficSign18.jpg "Traffic Sign 5"
[image19]: ./examples/TrafficSign19.jpg "Traffic Sign 5"
[image20]: ./examples/TrafficSign20.jpg "Traffic Sign 5"
[image21]: ./examples/TrafficSign21.jpg "Traffic Sign 5"
[image22]: ./examples/TrafficSign22.jpg "Traffic Sign 5"
[image23]: ./examples/TrafficSign23.jpg "Traffic Sign 5"
[image24]: ./examples/TrafficSign24.jpg "Traffic Sign 5"


---
### Data Set Summary & Exploration

Data set consists of 34K training samples, 4.4K validation and 12.6k. There are 43 distinct traffic signs in the dataset. 

Dataset Summary

Number of training examples      = 34799 (67%)
Number of validation examples    = 4410  (8.5%)
Number of testing examples       = 12630 (24.36%)
Image data shape = (32, 32, 3)
Number of classes = 43

Let’s examine few of the traffic sign classes:

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Observations

We can observe that traffic signs of the same appear differently due to following aspects
*	Lighting conditions
*	Brightness
*	Angle or position
*	Sharpness
*	Size
*	Color

It will be interesting to observe the distribution of the traffic sign classes.

### Traffic Sign Distribution

![alt text][image4]

### Top 10

![alt text][image5]

Speed Limit 50 kmph has the highest number of example images (2000).

### Bottom 10

![alt text][image6]

Speed Limit 20kmph has the lowest number of example images (~ 180).

Summary Statistics

![alt text][image7]

Given the that lower quartile has only 285 images, we can augment the number of examples for these classes by faking the data. I applied rule of augmenting the traffic sign class by adding 400 images per class for the signs with less than 1000 example images. Initially I had planned to augment all the classes but the net result could be that the distribution would remain the same. Combinations of random rotation, translation, 
sharpening or histogram equalization.

### Data Augmentation

Given the imbalanced data set, we can augment the dataset by generating fake images. Images can be generated through following operations:
*	Rotation [-15, 15]
*	Translation [-2, 2]
*	Histogram Equalization – it is technique for adjusting image intensities by enhancing the contrast
*	Sharpening

I have generated fake images by transforming the images through series of few of the above set of operations. Below are few examples of such augmented images:


![alt text][image8] ![alt text][image9]
Original Image       Augmented Image

![alt text][image10] ![alt text][image11]
Original Image       Augmented Image

### Data Pre-processing

Below were the data pre-processing tests done:

a)  Convert to gray scale

 Color Image         Grayscale Image

![alt text][image12] ![alt text][image13]


b) Normalize data set to be in the range of [0, 1]

As per the Le-Net 5 implementation, color channels would not help in increasing the accuracy, which was my experience as well. Thus, 
gray scale conversion was applied. Further I have experimented with pixel ranges such as -5 to +5, -1 to +1 but found that 
simple division by 255 yielding a [0,1] range results in better accuracy. 

Since in each layer we do dot products with the weights on the input vectors and with deeper networks, it’s possible that the gradients 
might explode. Thus, normalizing the data to a fixed range may avoid such explosion of gradients.


### Model Architecture


![alt text][image14] 

Figure: 6-layer Convolutional Neural Network

#### 6 – Layer Convolutional Network

![alt text][image15] 

### Fine tuning

I have experimented with various values for Learning rate, number of epochs, droput and number of layers as shown below:

![alt text][image16] 

Eventually, Learning rate of 0.0003, number of epochs = 50, droput percentage of 0.8 and 6-layer network provided the best performance. 


### Training, Testing and Validation

#### Results

Using the above parameters above, my validation accuracy is 96.8% and testing accuracy is 95.6%.

a) Without Data Augmentation

![alt text][image17] 

a) With Data Augmentation

![alt text][image18]

Data augmentation does not seem to particularly improve the performance. Perhaps a different combination of image transformations or lesser 
number of transformed images per traffic sign might help in improving the accuracy.

### Training vs Validation Accuracy

Below is the plot for training vs validation accuracy:

![alt text][image19]

### Predicting Sign of New Images

I have selected below set of random images from the web and tested in on my network:

![alt text][image20]

First image seems to be difficult to classify as casting of shadow seems to suggest that the traffic sign is End of Speed Limit 80 
instead of traffic sign speed limit 80. Also, another curve ball for the network is traffic sign speed limit 60. It almost looks 
like speed limit 80, which is even confusing for the human 😊 As anticipated, network fails to recognize this image as speed limit 60. 
Despite the brightness or the lack of it, network manages to recognize the images as speed limit sign 70. Next session displays the 
predicted sofmax probabilities of the network.

### Top 5 Softmax probabilities

![alt text][image21]
![alt text][image22]
![alt text][image23]

Model could correctly classify 7/9 images correctly resulting in a 77.78% accuracy. This is lower than that of Test set because of 
the smaller sample size and curve ball images in the test set. Looking at the softmax probabilities it is quite evident that model 
is quite confident even when making mistakes.

### Visualizing the Network

I have tried to visualize the features learnt by the network in each layer. At the time of writing, author is facing issues with 
manipulating tensors to generate the visuals. Expecting to fix this issue soon.
