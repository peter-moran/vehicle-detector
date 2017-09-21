# Vehicle Detection Project Write-up

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I desired to keep my feature extraction method very clean and programmatic so that I could confidently combine any number of features into one feature vector, do so on a lot of samples at a time, and do so the same way every time without having to re-write code.

As a result, I created a set of functions in `feature_extraction.py`, all used specifically for building one single feature vector for any given image. The primary interface to this is the `CarFeatureVectorBuilder` class, which defines the feature vector building pipeline for this project and was tuned for the best performance. It works by running the `generate_feature_vectors()` function, which accepts a list of functions (such as a HOG extractor function or a color histogram function) and uses them to build the entire feature vector.

`CarFeatureVectorBuilder` is also designed so that it can calculate HOG features for you (as typical during training) or you can pass in pre-calculated HOG features (as used during sliding window search to save on computation). Either way, `CarFeatureVectorBuilder` guarantees they will be treated the same, as long as any pre-calculated HOG features are computed by `car_feature_vector_builder.hog_features(img)`.

Ok, with the architecture explained, let's talk about how we actually obtain the HOG features. HOG features are obtained from [`get_hog_features()`](https://github.com/peter-moran/vehicle-detector/blob/a73846e103bc338a0bb43e45a91c92e87ea884a9/feature_extraction.py#L30) in `feature_extraction.py`, which performs standard HOG extraction using `skimage.feature.hog()`, but on multiple color channels and with optional color space conversion.

As mentioned earlier, all parameters for feature extraction are found in `CarFeatureVectorBuilder`. Here you will find [`self.hog_param`](https://github.com/peter-moran/vehicle-detector/blob/a73846e103bc338a0bb43e45a91c92e87ea884a9/feature_extraction.py#L187), which shows that the HOG features are computed with 8x8 pixel cells, 9 orientations per cell, and is normalized over 2x2 cell blocks. In addition, the color space is converted from RGB to YCrCb.

####2. Explain how you settled on your final choice of HOG parameters.

With a well structured pipeline set up, I simply tried a few different parameters in order to get the best test accuracy. I also found that I needed to do additional tuning in order to improve performance on the test video, so I would both extract features, train the SVC, and then try it on the video until I got the performance I liked.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained my SVM in `train.py`.

Specifically, I took about 1400 samples from both datasets, extracted the features using my `CarFeatureVectorBuilder`, and normalized across features using a sklearn StandardScaler.

Now with my feature vectors in `X` and my labels in `y`, I split the samples into a train and validation set, with 30% of the samples reserved for validation.

Next, I performed a random parameter search for an rbf SVM. This runs the SVM with various different settings for the `C` and `gamma` hyper parameters in order to find the combination with the best performance. The best classifier used `C=65.1` and `gamma=.0000882` and has a 89.2% accuracy when tested on a subset of unique images I selected using `clean_dataset.py` (more info in README).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in [`CarFinder.window_search_cars()`](https://github.com/peter-moran/vehicle-detector/blob/a73846e103bc338a0bb43e45a91c92e87ea884a9/find_cars.py#L155) which is [called over multiple scales](https://github.com/peter-moran/vehicle-detector/blob/a73846e103bc338a0bb43e45a91c92e87ea884a9/find_cars.py#L110) by `CarFinder.find_cars()`.

I chose the scales to search by testing out multiple values. I started with large windows, and then added more as needed to detect at smaller scales, only adding more when needed. To save on computation, I tried to minimize the y search range and overlap as much as possible.

Ultimately, I chose to do three passes:

|              | y-range | Window Scale | Window Overlap |
| ------------ | ------- | ------------ | -------------- |
| **Search 1** | 380-500 | 1            | 6/8            |
| **Search 2** | 380-550 | 1.3          | 5/8            |
| **Search 3** | 380-600 | 2.2          | 6/8            |



Below is an example of window selection in a single frame. The blue boxes are the selected windows and the green box is where the car is estimated to be based on those windows.

 ![windows](data/documentation_imgs/windows.png)

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=GmLn8OzekBA) and a [diagnostic view](https://www.youtube.com/watch?v=wKw9EWHOrDI) which shows the window detections and heatmap too.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

![car-diagnostic](data/documentation_imgs/car-diagnostic.gif)

The clip above comprises all aspects of filtering to true positives. The main loop for this filtering is in `CarFinder.find_cars()`. Let's explain:

Every window detection has a corresponding "score", or you might call it a "heat". This score is the confidence the SVM has in the window, measured by `clf.decision_function(X)`. When we perform a window search in `CarFinder.window_search_cars()` we return the location of the windows with cars in them, as well as the scores for these windows.

We perform the window search on three scales, as discussed. Next, we add all of the areas covered by each window to a single image, each window contributing it's score. This is done by `gen_heatmap()`. Next, we limit the maximum heat by setting any pixel above `CarFinder.max_frame_heat` to that max frame heat. 

Next, we add that heatmap for the single frame to a circular queue of the last 16 frames' heat maps.

To select for cars, we combine the last 16 frames into one singular heatmap, which is represented in the faded red squares in the clip above.

The combined heatmap is then thresholded, removing any pixels with an intensity below `heat_thresh_low`. Then we label the heatmap using `scipy.ndimage.measurements.label()`, which shows which pixels are connected on the heat map, giving each "island" its own label number.

Next, we find the bounding box that contains all the pixels in each labeled region. When we do this, we also check the intensity of the heatmap at the very center of that bounding box. If the core temperature is below `heat_thresh_high` then the label region is not used. This has a similar effect to hysteresis thresholding, and is performed in the line:

```
bboxes = hot_label_regions(labels, heatmap_thresh, threshold=heat_thresh_high)
```

These `bboxes` are the bounding boxes surrounding each car detection, and are then displayed on the image.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I managed to solve was dealing with false positives. At first I tried using plain old thresholding, where a region was identified as having a car in it if a car was detected there the last X% of frames, which didn't always work. Utilizing `clf.decision_function(X)` took me a step further, though, because it conveyed a measure of confidence in each measurement. However, what finally worked for me was when I emulated the behavior of hysteresis thresholding. I noticed that the heatmap was always very strong in the center of true car detections, but if I were to threshold for that value alone, the window would appear too small. So instead, I looked at the center of each label and checked if the center heatmap intensity was strong enough. If it was, I considered it (and the rest of the label) a car. This worked really well at detecting cars, keeping the whole area covering the car, and ignoring false detections.

The biggest problem I had was the intense amount of time the SVC took to run. Even using a linear kernel, the call to the SVM classification would take up over 50% of the program's runtime alone. In the future, I would consider using a neural network, as GPU acceleration might be really useful here and could help reduce the number of false positives more.

