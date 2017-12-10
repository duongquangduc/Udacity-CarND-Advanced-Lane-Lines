import cv2
import numpy as np
import matplotlib.image as mpimg

def toBinary(image, r_thresh=(210, 255), s_thresh=(120, 255), sx_thresh=(20, 100)):
    image = np.copy(image)

    R = image[:,:,0]
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1 
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, r_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(r_binary == 1) | (sxbinary == 1)] = 1   
    return combined_binary
    
# # Define a function to return the magnitude of the gradient
# # for a given sobel kernel size and threshold values
# def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 255)):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # Take both Sobel x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

#     # Calculate the gradient magnitude
#     gradmag = np.sqrt(sobelx**2 + sobely**2)

#     # Rescale to 8 bit
#     scale_factor = np.max(gradmag)/255
#     gradmag = (gradmag/scale_factor).astype(np.uint8)

#     # Create a binary image of ones where threshold is met, zeros otherwise
#     binary_output = np.zeros_like(gradmag)
#     binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

#     # Return the binary image
#     return binary_output

# # Define a function to threshold an image for a given range and Sobel kernel
# def dir_threshold(img, sobel_kernel=15, thresh=(0, np.pi/2)):
#     # Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # Calculate the x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

#     # Take the absolute value of the gradient direction,
#     # apply a threshold, and create a binary image result
#     absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
#     binary_output =  np.zeros_like(absgraddir)
#     binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

#     # Return the binary image
#     return binary_output

# def color_threshold(img, r_thresh=(0, 255), s_thresh=(0, 255)):
#     # Apply a threshold to the R channel
#     r_channel = img[:,:,2]
#     r_binary = np.zeros_like(img[:,:,0])
    
#     # Create a mask of 1's where pixel value is within the given thresholds
#     r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

#     # Convert to HLS color space
#     hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
#     # Apply a threshold to the S channel
#     s_channel = hls[:,:,2]
#     s_binary = np.zeros_like(s_channel)
    
#     # Create a mask of 1's where pixel value is within the given thresholds
#     s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

#     # Combine two channels
#     combined = np.zeros_like(img[:,:,0])
#     combined[(s_binary == 1) | (r_binary == 1)] = 1
    
#     # Return binary output image
#     return combined
