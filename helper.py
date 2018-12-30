import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# prepare objpoints and imgpoints for undistortion
def findImgsPoints(filenames, nx=9, ny=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    # np.mgrid[0:9,0:6] => (2,9,6) Array
    # np.mgrid[0:9,0:6].T => (6,9,2) Array
    # np.mgrid[0:9,0:6].T.reshape(-1,2) =>(54,2) Array

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(filenames)

    # Step through the list and search for chessboard corners
    for fname in images:
        image = mpimg.imread(fname)
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            #img = cv2.drawChessboardCorners(image, (9,6), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(600)
            
            #cv2.destroyAllWindows()
    
    return objpoints, imgpoints


# undistort color img
def undistort(image, mtx, dist):
  
    #gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # for cv2.cvtColor()，the color channel is BGR, not RGB
      
    # using cv2.undistort() undistort the picture
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist

# select hls channel
def hls_select(image, channel= 'S', thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if channel == 'S':
        slt = hls[:,:,2]
    if channel == 'H':
        slt = hls[:,:,0]
    if channel == 'L':
        slt = hls[:,:,1]
   
    binary_output = np.zeros_like(slt)
    binary_output[(slt>thresh[0])&(slt<=thresh[1])] = 1
    
    return binary_output

# sobel_x and sobel_y	
def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh=(0,255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = img
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    elif orient =='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    else:
        raise Exception("Invalid orient!")
        
    abs_sobel = np.absolute(sobel)
    
    # scaled_sobel = np.unit8(255*abs_sobel/np.max(abs_sobel)) *report numpy no unit8 attribute error*
    scaled_sobel = np.array(255*abs_sobel/np.max(abs_sobel),dtype=np.uint8)
    #scaled_sobel = np.unit8(255*abs_sobel/np.max(abs_sobel))
        
    binary_output = np.zeros_like(scaled_sobel)
    
    # binary_output[(scaled_sobel>thresh_min)and(scaled_sobel<thresh_max)] = 1 *report error
    binary_output[(scaled_sobel>thresh[0])&(scaled_sobel<thresh[1])] = 1
         
    return binary_output

# magnitude threshold sobel 
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobel = np.sqrt(abs_sobelx*abs_sobelx+abs_sobely*abs_sobely)
    
    scaled_sobel = np.array(255*abs_sobel/np.max(abs_sobel),dtype=np.uint8)
    
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>thresh_min)&(scaled_sobel<thresh_max)] = 1
    
    return binary_output

# direction threshold sobel
def dir_threshold(img, sobel_kernel=3, thresh=(np.pi/6, np.pi/3)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    theta_mtx = np.arctan2(abs_sobely,abs_sobelx)
    
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output = np.zeros_like(theta_mtx)
    binary_output[(theta_mtx>thresh_min)&(theta_mtx<thresh_max)] = 1
    
    return binary_output

# combined gradient and direction threshold
def combined_thresh(img,ksize=5, grad_thresh=(0, 255), dir_thresh=(np.pi/6, np.pi/3)):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=grad_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(dir_binary)
    combined[(mag_binary==1)&(dir_binary==1)] = 1
    return combined

# perspective view	
def corners_unwarp(undist,src,dst,img_size):
   
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M
	

# mask interested binary area
# function name should be not same as return name
def binary_masked(binary, vertices):
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [vertices], 255)
    binary_mask = cv2.bitwise_and(binary, mask)
    return binary_mask

#histogram of the bottom half of a warped binary image
def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half,axis=0) # axis=0, 行向量相加
    
    return histogram
	
# sliding windows to hist the warped binary image
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = hist(binary_warped)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # good_left/right_inds 实际是一个1D数组；
        #它利用了nonzerox和nonzeroy中的同一index的元素表示同一个点的性质
        # 下式中的逻辑表达式表示了nonzerox和nonzeroy中落在窗口中的点的个数
        # 然后通过nonzero()返回了1D数组中非零值的index，该index同时用于nonzero(index)和nonzero(index)
        # 从而得到落到该窗口的某个像素点的坐标
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        #pass # Remove this when you add your function
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img
	
# Fit a Polynomial of the sliding window filtered image
def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    if (len(leftx)>0)&(len(lefty)>0)&(len(rightx)>0)&(len(righty)>0):
        left_fit = np.polyfit(lefty,leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
    else:
        left_fit = np.array([1.0,1.0,1.0])
        right_fit = np.array([1.0,1.0,1.0])

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, ploty, left_fit, right_fit

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    # left_fitx, right_fitx 通过拟合公式得到的x值
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx,left_fit, right_fit, ploty

# seach around the fitted poly
def search_around_poly(binary_warped, left_fit, right_fit, margin = 100):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    #margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    
    # Again, extract left and right line pixel positions
    # 因为所有区域内的像素点是一次取出，所以可以通过逻辑列表来得到对应的X坐标
    # 因为同一像素点在nonzerox和nonzeroy中表示为同一index的元素，所以也可得到对应的Y坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    if (len(leftx)>0)&(len(lefty)>0)&(len(rightx)>0)&(len(righty)>0):
        left_fitx, right_fitx,left_fit, right_fit, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    else:
        out_img, ploty, left_fit, right_fit = fit_polynomial(binary_warped)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, ploty, left_fit, right_fit


def measure_curvature_pixels(left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####

    ## Implement the calculation of the left line here
    left_curverad = (1 + (2*left_fit[0]*np.max(ploty)+left_fit[1])**2)**1.5/np.abs(2*left_fit[0])  
    ## Implement the calculation of the right line here
    right_curverad = (1 + (2*right_fit[0]*np.max(ploty)+right_fit[1])**2)**1.5/np.abs(2*right_fit[0])  
    
    return left_curverad, right_curverad
	

def measure_curvature_real(left_fit, right_fit, ploty, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
   
    # rectify the polyfit coefficient to real
    fit_per_pix = np.array([xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix])
    left_fit_cr = fit_per_pix*left_fit
    right_fit_cr = fit_per_pix*right_fit
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image

    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    ## Implement the calculation of the left line here
    left_curverad = (1 + (2*left_fit_cr[0]*np.max(ploty*ym_per_pix)+left_fit_cr[1])**2)**1.5/np.abs(2*left_fit_cr[0])
    ## Implement the calculation of the right line here	
    right_curverad = (1 + (2*right_fit_cr[0]*np.max(ploty*ym_per_pix)+right_fit_cr[1])**2)**1.5/np.abs(2*right_fit_cr[0])  
    
    return left_curverad, right_curverad
	

def car_offset(binary_warped, left_fit, right_fit, xm_per_pix):

    center = binary_warped.shape[1]//2
    bottom_y = np.array([(binary_warped.shape[0])**2, binary_warped.shape[0], 1])
    lane_right = right_fit.dot(bottom_y)
    lane_left = left_fit.dot(bottom_y)
    lane_mid = np.int32((lane_right + lane_left)/2)
    lane_width = np.int32(lane_right - lane_left)
    offset = (center - lane_mid)*xm_per_pix
    lane_width_cr = lane_width*xm_per_pix
    return offset,lane_width_cr,lane_width
	
	
def draw_lane_window(img, warped, src, dst, left_fit, right_fit, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, color_warp[:,:,0].shape[::-1], flags=cv2.INTER_LINEAR) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result
	
	

