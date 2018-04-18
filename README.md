
**Vehicle Detection**

The goals / steps of this project are the following:

* Perform Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalizing features and randomize selection for training and testing.
* Implement sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output-image/test-image.png
[image2]: ./output-image/HOG-feature.png
[image3]: ./output-image/visualizatoin-original.png
[image4]: ./output-image/noncar-HOG.png
[image5]: ./output-image/not-best.png
[image6]: ./output-image/slide.png
[image7]: ./output-image/one.png
[image8]: ./output-image/heat.png
[image9]: ./output-image/before.png
[video1]: ./output-video/lane_car.mp4
[video2]: https://www.youtube.com/watch?v=ApZARQY--6U&feature=youtu.be

---

### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images.

The HOG feature extraction is necessary for every frame process and it is implemented as below:

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

Higher orientation parameters leads to longer training time but not necessarily better results in terms of accuracy. I tried `orientations` from 8 to 10 and found 9 gives relatively better results and maintains a slightly less training time.

I also tried increasing and decreasing the `pixels_per_cell` and `cells_per_block`. When using `pixels_per_cell` at 7 or 9, no siginificant improvements can be observed while `cells_per_block=(1, 1)` produces less accurate results.

![alt text][image5]

#### 3. Classifier Training

The training is done with sklearn with the below code:

```python
# Use a linear SVC
clf = LinearSVC()
# Check the training time for the SVC
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
```

Prior to the training, input features are also scaled with `StandardScaler()`

```python
X_scaler = StandardScaler().fit(X)
X_all = X_scaler.transform(X)
```

### Sliding Window Search

#### 1. Implementation, scale and overlapping ratio

The sliding window size is 64 by 64 and 2 pixel per step. As for scaling, I found when using only one value, sometimes there will only be a single box detected and thus cannot create a "hot" region and later be eliminated by false positive filter. Thus I used a range of `1.5 - 2` for scaling

![alt text][image6]

The "only one box detected" problem can be visualized as below:

![alt text][image7]


#### 2. Example image demonstration

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image9]
---

### Video Implementation

#### 1. Provide a link to your final video output.
[GitHub link](./project_video.mp4)

[YouTube link](https://www.youtube.com/watch?v=ApZARQY--6U&feature=youtu.be)

#### 2. False positive and 

I recorded the positions of all bounding boxes in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image1]


---

### Discussion

#### 1. Problems and future plans

The bounding boxes are not stable in the output video. A stablizer which takes the previous frames and bounding box center locations and outputs the incremental bounding box of the current frame can be used to solve this.
