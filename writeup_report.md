## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

## Writeup Report / README

This is the project 4: Advanced Lane Finding in Self-Driving Car Nanodegree course by Udacity. The goal is to write a software pipeline to identify the lane boundaries in a video.

For the original assignments can be found in [the project repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

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

[image1]: ./output_images/undistorted_image.png "Undistorted"
[image2]: ./output_images/undistorted_images.png "Undistorted-Calibration"
[image3]: ./output_images/thresholded_binary_image.png "Thresholded Binary Image"
[image4]: ./output_images/perspective_transform_image.png "Perspective Transform Image"
[image5]: ./output_images/lane_boundary.png "Radius Curvature"
[image6]: ./output_images/radius_curvature_formula.png "Radius Curvature Formula"
[image7]: ./output_images/curvature.png "Curvature"
[image8]: ./output_images/warpBack.png "Warp the detected lane boundaries back onto the original image"
[image9]: ./output_images/writetext.png "Display Curvature and Vehicle Position"

[video1]: ./project_video_result.mp4 "Video"

### Project Structure

My project includes the following files:
* Advanced_Lane_Finding.ipynb file containing all the steps performed in the project.
* writeup_report.md file containing the explanation for each steps
* utils is the folder for the python files like thresholds.py, etc...
* output_images folder for all the output images for the images in camera_cal and test_images folders.
* project_video_result.mp4 showing the final result for lane finding
* report.pdf is the pdf file of Advanced_Lane_Finding.ipynb
* Other default files in the project repository 

---

### Writeup / README

### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

The code for this step is contained in the code cells of the IPython notebook located in "./Advanced_Lane_Finding.ipynb".
The main process contains the following steps:

Step 1: Preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. 

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

Step 2: Computing the camera calibration and distortion coefficients by using the output `objpoints` and `imgpoints` and `cv2.calibrateCamera()` function.
`
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
`

Step 3: Applying the distortion correction the `cv2.undistort()` function to the test images. Here is an example of the result:
`
def undistort_image(img):
    # Undistort a test image
    img = cv2.undistort(img, mtx, dist, None, mtx)
    return img
`

![alt text][image1]

In the first part, I also draw and display the chess corners. Please refer to the jupyter notebook for the display.

### 2. Apply a distortion correction to raw images

I applied the function `undistort_image()` to the image samples in camera_cal and test_images folders and save the results into ./output_images/ folders.
Please check the folders in Github to verify.

![alt text][image2]


### 3. Use color transforms, gradients, etc., to create a threshold binary image
	
I used a combination of color and gradient thresholds to generate a binary image. The `toBinary()` funtion is defined in ./utils/thresholds.py file.
The final image is shown as below.

![alt text][image3]

### 4. Apply a perspective transform to rectify binary image ("bird-eye-view")

The code for my perspective transform includes a function called `warper()`, which appears in line [8] in the IPython notebook.
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source 
and destination points in the following manner:

```python
s# offset for dst points
    offset = 350
    
    # Source points
    src = np.float32([[[ 610,  450]], 
                      [[ 680,  450]], 
                      [[ img_size[0]-300,  680]],
                      [[ 380,  680]]])

    # Result points
    dst = np.float32([[offset, 0], 
                    [img_size[0]-offset, 0], 
                    [img_size[0]-offset, img_size[1]], 
                    [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 610, 450      | offset, 0        | 
| 680, 450      | img_size[0]-offset, 0     |
| img_size[0]-300,  680     | img_size[0]-offset, img_size[1]      |
| 380, 680      | offset, img_size[1]        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]


### 5. Detect lane pixels and fit to the lane boundary

I used a convolution function `find_window_centroids` which will maximize the number of pixels in each window. A convolution is the summation of the product of two seperate signals,
in this case, they are the window template and the vertice slice of the pixel image.

By slidingthe window template across the image from left to right, the overlapping values are summed together, created the convolved signal. The peak of the colvolved signal is where
there was the highest overlap of pixels and the most likely position for the lane marker.

Finally, combining the left and right points over the original image. Here is the example:
![alt text][image5]


### 6. Determine the curvature of the lane and vehicle position with respect to center

The radius of curvature can be calculated as the following equation:
![alt text][image6]

By using the `leftx` and `rightx` from the `find_window_centroids`, I can define the funtion `curvature()` as in line [11].
![alt text][image7]

### 7. Warp the detected lane boundaries back onto the original image

I implemented this step in lines #12.  Here is an example of my result on a test image:
![alt text][image8]

### 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
I used text() from matplotlib, to write the numerical estimation of lane curvature and vehicle position on top of the image.
Here is one example:
![alt text][image9]

### 9. The Final Pipeline
Combining all the steps from the previous, I define `lane_finding()` function as in line #14.

The I applied the funtion for project video in step #10 and see it gets a very good result.

---

### 10. Pipeline (video)

#### Please check my output video for the project video as below:

Here's a [link to my video result](./project_video_result.mp4). Or you can watch it by clicking the below thumbnail.

[![Alt text](http://img.youtube.com/vi/LvZqJqIDDBo/0.jpg)](https://youtu.be/LvZqJqIDDBo)

---

### Discussion

The final `lane_finding()` funtion works pretty well in the project_video but show limited results in the challenge_video and harder_challenge_video. It may be due to different light conditions, the huge curvatures or speed...So I need to find further advanced techniques to overcome these causes.

There are many things that I could improve this projects like:
* Using different transforms techniques (colors, gradients, etc...) to covert an image to its binary image and compare the performance on the algorithms.
* Experiment with different threshold values, color spaces, its combinations, etc...
* Improve the perspective transform to rectify binay image by applying different techniques
