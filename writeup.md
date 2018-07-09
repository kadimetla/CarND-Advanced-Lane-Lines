## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_test_image1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/test1.jpg "Binary Example"
[image4]: ./output_images/test5.jpg "Warp Example"
[image5]: ./output_images/lane_lines_image1.jpg  "Fit Visual"
[image6]: ./output_images/lane_lines_area_image1.jpg  "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to compute camera matrix and distrontion coefficeints is in IPython notebook located in "camera_calibration.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I saved the results camera matrix and distortion coefficients to "camera_cal/dist_pickle.p" as numpy object serializtion pickle file format. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

1. read the pickle file "camera_cal/dist_pickle.p" 
2. read camera matrix and distortion coefficients from the pickle file
3. start reading test images and apply undistory method cv2.undistort.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I did lot of playing with colors and gradients in it's  own IPython notebooks.

1. color transformation I played with colors in "play_color_thresholds.ipynb" and coded the methods in      advanced_lane_finding.ipynb. code could be organized better.

2. gradient thresholds I played in "play_magnitude_thresholds.ipynb". I exported the file and used as python file "magnitude_thresholds.py" code.


I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at method color_and_gradient in `advanced_lane_finding.ipynb`).  Here's an example of my output for this step. test images with pipeline original image -> undistored image -> color and combined managnitude -> binary warped -> lane lines using histogram and polynomial -> lane lines area drawn on the image.
I had problem writing so I saved as figure. which contains entire pipe line of transformation.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(image)`, which appears in IPython notebook 'advanced_lane_finding.ipynb'. The `perspective_transform(image)` function takes as inputs an image (`image`), source (`src`) and destination (`dst`) points are defined in the function.  I computed source and destination points using precentages as shown the advanced lane line video project. I did try hardcode source and destination. I am having problem with coming with right  source and destination points for perspective transformation and warped binary image.

```python
src = np.float32([[545, 460],
                    [735, 460],
                    [1280, 700],
                    [0, 700]])
    
    dst = np.float32([[0, 0],
                     [1280, 0],
                     [1280, 720],
                     [0, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545, 460      | 0, 0        | 
| 735, 460      | 1280, 0      |
| 1280, 700     | 1280, 720      |
| 0, 700        | 0, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code handle is coded in class LocateLaneLines. Which is used in the pipe line. 
Then I did histogram and fit my lane lines with a 2nd order polynomial using sliding window kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in class LocateLaneLines to compute using measure_lane_lines_curvature_in_real_world_space in the IPython notebook file `advanced_lane_finding.ipynb`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in IPython notebook file `advanced_lane_finding.ipynb` in the function `undistorted_image_with_lane_area_drawn`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [https://s3-us-west-2.amazonaws.com/udacity.selfdrivecar/P41/project_video.mp4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

pipe line fails during the curves, as I did not smooth the lane lines from previous frames. I also feel binary warped images are also contributing to the cause.

Below points are important, I have to do them for smooth lane flow.

1. **Sanity Check**

Ok, so your algorithm found some lines. Before moving on, you should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider:

Checking that they have similar curvature
Checking that they are separated by approximately the right distance horizontally
Checking that they are roughly parallel

2. **Look-Ahead Filter**

Once you've found the lane lines in one frame of video, and you are reasonably confident they are actually the lines you are looking for, you don't need to search blindly in the next frame. You can simply search within a window around the previous detection.

For example, if you fit a polynomial, then for each y position, you have an x position that represents the lane center from the last frame. Search for the new line within +/- some margin around the old line center.

Double check the bottom of the page here to remind yourself how this works.

Then check that your new line detections makes sense (i.e. expected curvature, separation, and slope).

3. **Reset**

If your sanity checks reveal that the lane lines you've detected are problematic for some reason, you can simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for several frames in a row, you should probably start searching from scratch using a histogram and sliding window, or another method, to re-establish your measurement.

4. **Smoothing**

Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.
