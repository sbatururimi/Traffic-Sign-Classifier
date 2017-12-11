# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_reports/samples.png "Visualization"
[image2]: ./images_reports/distribution.png "Distribution"
[image3]: ./images_reports/grayscale.png "Grayscaling"
[image4]: ./images_reports/perspective_distortion.png "Perspective distortion"
[image5]: ./images_reports/transformation_scale.png "Scaling transformation"
[image6]: ./images_reports/several_transformations.png "Combining transformations"

[image7]: ./data_web/image_0.jpg 
[image8]: ./data_web/image_1.jpg 
[image9]: ./data_web/image_2.jpg 
[image10]: ./data_web/image_3.jpg 
[image11]: ./data_web/image_4.jpg 

## Rubric Points
<!-- ### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) -->

We first start by exploring our dataset and check wheta kind of images we have:
![Initial dataset][image1]

We need to understand how normalized is the distribution of samples across each classes. 
![Dataset distribution][image2]

The distribution graphic clearly shows that several traffic classes are less represented in our dataset. There are a lot of different technics used in the industry in order to augment the dataset, like affine transformatiosn, distortions, adding noises (gaussian noise, salt and pepper, motion noise, poisson noise, etc.). According to ["The Effectiveness of Data Augmentation in Image Classification using Deep Learning"](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf), we could use different approaches to augment the dataset like CycleGans, an augmentation network but traditional transformations that consist of using a combination of affine transformations to manipulate the training still give us a good result in order to prevent an underfitting.

For simplicity we use brightness modifications and perspective distortion mostly
![Perspective transformation & brigntness][image5]

while experimenting with salt & pepper noise and scaling transformation. The last two methods were removed from the end work in order to speed up the training process. We generate around 5000 samples for each of the 43 traffic sign classes.
![Scaling][image6]

### Training & evaluation
The primary architecture was based on LeNet architecture and directly gave us a validation accuracy of ~89%. In order ot improve that we added more layers, a dropout as regularization in order to prevent an overfitting and finally after several tests we introduced batch normalization in order to normalize the inputs to layers within the network. Using such technics helped to get an accuracy of ~96% on the validation set after 50 epochs and an accuracy of~95% on the test set.


### Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic
signs data set:
Prior to the augmenattion:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3), i.e an RGB image
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was initialy distributed before a data augmentation was applied to it

![Distribution][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this is generally how impage processing work. What is important here, is that I believe using all 3 channels of the RGB channels would lead to better results. In "Systematic evaluation of CNN advacnces on the ImageNet" research paper, most of the evaluation are done without converting the RGB image to a grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][image3]

We could apply other preprocessing technics in order to remove different noises we could expect from samples. Examples of smoothing technics can be find [here](https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html)

As a last step, I normalized the image data because if we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be over compensating a correction in one weight dimension while undercompensating in another.

I decided to generate additional data because some classes are less represented and this may lead to low accuracy on the training and validation sets imply underfitting.

To add more data to the the data set, I used the following techniques because generating a wide range of different images:
 - brightness modifications 
 - perspective distortion

Here is an example of an original image and an augmented image:

![Augmentation][image6]

The difference between the original data set and the augmented data set is the following:
- each class has been augmented in order to have 5000 samples
- for each batch we take around 16% of the initial set and the remaining part is provided by the augmented set. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input 				| 32x32x3 RGB image   							| 
| Preprocessing	        | 32x32x1 Gray image							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Batch normalization   | 												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| Batch normalization   | 												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Fully connected		| output  800  									|
| Fully connected		| output  256  									|
| Fully connected		| output  84  									|
| dropout 				|												|
| Softmax				| 		       									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an an Amazon GPU instance. As optimizer, Adam optimizer was used. The batch size was equal to 128, epoch = 100 and the keep dropout was of 0.7. 
The learning rate finally choosed was equal to 0.001 while some expirements was made with linear learning rate policy. It was abandoned due to the required time necessary to train the network with such policy which was much higher than when a constant was used..

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96,6 
* test set accuracy of 95,1

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was LeNet architecture without using dropout nor batch normalization.
* What were some problems with the initial architecture?
It couldn't go higher than ~89%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
In order to prevent an overfitting dropout was added to the fully connected layer also batch normalization helped during training, as we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch. This allowed higher learning rates and the network trained much faster.
Also due to the big amount of datas, batches were previously saved inside files loaded during the training process.
* Which parameters were tuned? How were they adjusted and why?
The learning rate was adjusted a little as well as experiments with different epochs were done.
A popular learning rate is 0.001 or 0.01, while a linear learning rate policy were tested as recommended in "Systematic evaluation of CNN advacnces on the ImageNet" it was finally abandonned due to the necessity to train for hours on a GPU.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Because we are trying to predict the traffic sign on a image, Convolutional neural networks have been proven to perform well on such tasks while dropout  is a simple way to prevent neural networks from overfitting

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The architecture  is a classic convnet with dropout+batch noramlization performs well on classification problems linked to object detection in the image processing field.
We obtained an accuracy of ~95% on teh training set
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

The 4 & 5 images might be difficult to classify because the sign is small and we have removed the scaling step during the dataset augmentation

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Priority road      					| Priority road    								| 
| 80km.h  								| Yield 										|
| No entry								| No entry										|
| 120 km/h	      						| Bicycle crossing					 			|
| Right of way at the nextintersection	| Right of way at the nextintersection			|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the pre-last section of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority road (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road    								| 
| .0     				| Roundabout mandatory							|
| .0					| Yield											|
| .0	      			| Ahead only					 				|
| .0				    | Speed limit (50km/h)    						|


As expected, smaller traffic sign images gave a very low accuracy. This is visible for the 80km.h and 120 km/h. 

We could train the network with more modification linked to the scaling size as well as using a [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) is reported to give very high accuracy.



