##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/hog_vehicles_nonvehicles.png
[image31]: ./output_images/sliding_075.png
[image32]: ./output_images/sliding_100.png
[image33]: ./output_images/sliding_125.png
[image34]: ./output_images/sliding_150.png
[image4]: ./output_images/hotWindows_s.png
[image5]: ./output_images/heatmap_s.png
[image6]: ./output_images/resultingBound_s.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The implementation of the HOG extraction is implemented in the function `getHOGFeatures` which is a class member of VehicleDetector class. I've played around with the different colorspaces and came to the ideal combination for the given set of training-data: 

![alt text][image1]

I've taken the training samples offered [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and for non-vehicles [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

In order to do so, I've runned the LinearSVC on different combinations of Color-Histogram and HOG on different colorspaces. The table below show my favorites:

|       | RGB    | HLS    | HSV    | YCrCb  | LUV   | YUV    | Histogram
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RGB   | 0.9814 | 0.9868 | 0.9848 | 0.9842 | 0.984 | 0.9828 | 
| HLS   | 0.9868 | 0.991  | 0.991  | 0.9882 | 0.9896| 0.9885 |
| HSV   | 0.9848 | 0.9918 | 0.9887 | 0.9901 | 0.9913| 0.9893 |
| YCrCb | 0.9842 | 0.989  | **0.9938**| **0.9938** | 0.9916| 0.9873 |
| LUV   | 0.984  | 0.9916 | 0.9907 | 0.9893 | 0.991 | 0.9916 | 
| YUV   | 0.9828 | 0.9924 | 0.991  | 0.9913 | 0.9916| 0.9927 | 
| **HOG** |

The result is marked in **bold **: so either HSV or YCrCb in histogram features and YCrCb in HOG features.

The content of the table can be fetched by calling the function `compareColorSpaces` (about line 680).

During my first tries on the testimages I've already made very good results with the settings, so I didn't experiment with the other parameters like: orientations, pixel per cell or cells per block.

So my final setup was then:
- colorspace histogram features HSV, considering all three channels
- colorspace hog features YCrCb, considering all three channels
- hog orientation 9
- hog pixel per cell 8
- hog cells per block 2

Here is an example of the hog features using the configuration mentioned above on a window from the project video:

![alt text][image2]

I've written an own fuction `dumpHOGImage` which will create the picture above. In essence it uses the parameters mentioned above (which are stored as class members of the VehicleDetector class) and calling the member function `_getHOGFeatures` for all three channels of the image. The function `_getHOGFeatures` internally using `skiamge.feature._hog`.

Due to the fact that I was already making a very good detection with the choosen colorspaces and with the initial parameter mentioned above, I didn't try further combinations for HOG.

####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

During my first few experiments with the classifier i've played around with the LinearSVC and the SVC with a rbf kernel. As it was already mentioned in the lecture, the rbf kernel results in a better matching rate - considering the much larger runtime I decided to come as far as I can with a LinearSVC... and finally this works out for me.

For the training i've written a function `trainClassifier`. This method expects a path to all non-vehicle images as well as a path to the vehicle training samples. Internally it reads the images and applies the function `extractFeaturesGetScaler`.
The `extractFeaturesGetScaler` extracting the HOG and Histogram features from  all images. At the end, the features are scaled and normalized using the `sklearn.preprocessing.data.StandardScaler`. The result are the scaler itself as well as the scaled set of image-features.

Back in the `trainClassifier` method I split the set of features to a training and a verification set (using `train_test_split` function) and apply the `fit` function of the LinearSVC on the trainig data. As shown in the table above, my resulting accuracy on test-data is about 0.9938.




###Sliding Window Search

In order to process a whole video, a pipeline function `processImage` was written. This function does
* do a sliding window, extracting the HOG and Histogram features from the images (`findHotWindows` )
* merges the candidates of the sliding window results to a heatmap (`mergeHotWindow`)
* extracts the real detections (filtering false positives) after a certain set of frames

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First of all I've identified the part of the image i need to scan for vehicles. I chose the whole x scale but reduces the y scale from 400-640 px. The result is shown as the white rectangle in one of the images below.

Then instead of changing the searchWindow size which in the end had to result in a 64x64 px image (this is the shape of the training data used to train the SVC) i've decided to follow the approach from the lecture - changing image scale and keept the window size constant.

I've chosen 4 different scales [ 0.75, 1.0, 1.25 and 1.5] - the resulting window is then shown the images below as a blue rectangle.

![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]

The window were overlapping about 75%. The sliding window is implemented in function `findHotWindows`. HOG was applied to the whole image once while histogram feature extraction was applied from window to window. If my classifier predicts one of the windows to be a vehicle - see about line 430 - i've stored the window rectangle in a list. This list is returned to the `processImage` function.

After applying the sliding window in each scale I get a list of hot windows which looks e.g. like this

![alt text][image4]

As you can see, the resulting boundingboxex are matching very well on the white car while on the black car the boundingboxes are reaching to the road left to the car. In order to get rid of such false positives a heatmap was extracted from all given boundingboxes.

![alt text][image5]

The heatmap is created by counting the number of boundingboxes that overlap a certain pixel. At the end, all pixels below a certain threshold - in my case 4 - are nulled while the others kept as they are. The heatmap is identified in function `mergeHotWindows`. By combining pixels which are next to each other to boundingboxes, we receive the following result/ detection of vehicles

![alt text][image6]

Abviously still not very nice - conerning the white car which is detected as two seperated rectangles. The problem here is that some of the windows did not get enought heat and therefore get's discarded while comparing with the thresholde. Another problem I was facing was the detection of false positives - meaning windows which definetly didn't contain any vehicle but were predicted as vehicles.

In order to solve both problems, i've averages a bunch of subsequent frames - in this case the frame history was 8 (see `heatMapHistory` member). So what is happening during this averaging: i collect the hot windows of 8 frames and sum them up. Then i've adjusted the threshold to be original threshold multiplied with the frame history = 8. If a pixel is lower than this new threshold it must obviously be a false positive. Gaps in the bounds like shown in the picture above were disappering with this approach as well.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to project video result](./project_video_result.mp4) as well as to the [test video result](./test_video_result.mp4))



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project i've provided a pipeline which is using a linear SVC to identify vehicles in a video stream. The SVC is deciding based on HOG and Histogram features which were extracted from each video image in a sliding window approach. For the initial training of the SVC classifier i've used a set of about 8000 testsamples for vehicles and non-vehicles. 

The implementation is working very well on the given project video, but for sure there is many place for optimization.

First of all i'm really disappaointed of the runtime when processing the project video. This was talking me about 1 hour to process it. I've tried to speedup with a multithreaded approach but this leads not to the expected speedup. In order to run something like that in realtime, the critical path in this pipe must be identified and optimized.

During the first tests i've played around with the histogram distribution using the mean of all images. I was really suprised when i've figured out that the impact of adding / not using the histogram featues does only have a very small impact on the accuracy of the classifier. This leads to the following questions:

* could  the classifier only be tuned by the parameters of the HOG-extraction (ignoring the histogram features completely)?
* does the bining-features - i didn't use at all - provide any performance benefit for the classifier?

In essence to get a really reliable classifier, further test-samples must be used to verify and tune the classifier. 



