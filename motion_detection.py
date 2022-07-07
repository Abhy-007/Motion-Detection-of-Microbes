# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 00:33:31 2022

@author: 2014326
"""

# Importing necessary libraries
import cv2
from skimage import filters
from skimage.color import rgb2gray
import numpy as np

q = 12 #value that determines the size of the object to be identified

frame_ctr,fps,width,height = 0,0,0,0
input_path = "rawsample/ciliate.mp4" #path to the raw video file
output_path = "processed/ciliate.mp4" #path to the processed video file

cap = cv2.VideoCapture(input_path) #Storing the video inside an instance
if cap.isOpened(): #if the instance is opened then:
    width = cap.get(3) #get the width of the video frame inside the instance
    height = cap.get(4) #get the height of the video frame inside the instance
    fps = cap.get(cv2.CAP_PROP_FPS) # get the frame rate of the video inside the instance
fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #the format in which the processed output is needed

out = cv2.VideoWriter(output_path,fourcc,int(fps),(int(width),int(height))) #in what format the video has to be written
color = (0,0,255) # Color for drawing the boundary box
success = True #setting success parameter as true
img_old = None # setting image old parameter as None

while success: #When success is true -  this will make the loop iterate until the variable success = True
    success,im = cap.read() #Reading the video file - returns the whether it was successful in reading video and frame of the video
    if not success: 
        break
    im1 = np.copy(im) #making a copy of the frame returned by the cap.read()   
    if img_old is not None:

        
        image = np.copy(im) #making a copy of the frame returned by the cap.read()
        
        im = rgb2gray(im) # converting the frame into gray image

        img_old = rgb2gray(img_old) #converting the frame in img_old to gray image
        
        frame_difference = cv2.absdiff(im,img_old) #find the absolute difference between the present frame and previous frame
        
               
        #Find the indexes in frame difference which don't have 0 values - finding the values present in those indexes and taking the
        #mean of it- extracting those points in the frame_difference which is less than the collective mean and replacing it with 0
        #if it is greater then replacing the same with 1
        frame_difference[frame_difference < frame_difference[np.nonzero(frame_difference)].mean()] = 0
        frame_difference[frame_difference > frame_difference[np.nonzero(frame_difference)].mean()] = 1
        
        #finding the edge of the frame_difference image using sobel edge detector
        frame_differences = filters.sobel(frame_difference)
        
        #converting the frame_difference frame into 3 dimensions and stacking the values to each pixel and then *255
        frame_differences1 = np.stack((frame_differences,)*3,axis=-1)*255
        
        #converting the frame_difference1 image to gray color
        diff_image = cv2.cvtColor(frame_differences1.astype('uint8'), cv2.COLOR_BGR2GRAY)
        
        #taking threshold of the image        
        ret,thresh = cv2.threshold(diff_image,0,255,cv2.THRESH_BINARY)
        
        #finding the contours of the thresholded image
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #iterating through the detected contours and excluding those contours which are completely inside another contour
        for idx,c in enumerate(contours):
            if hierarchy[0][idx][3] ==-1:
                x,y,w,h = cv2.boundingRect(c)
                #checking if the detected bounding box area is less than the area of the object size declared. if yes the bounding
                #box won't be drawn
                if w*h <= q**2:
                    continue
                image = cv2.rectangle(image,(x,y),((x+w),(y+h)),color,3)
        out.write(image) #frame by frame the video is being written
    img_old = im1

    frame_ctr = frame_ctr+1
    if frame_ctr % 30 == 0:
        print("Frames processed so far: ", frame_ctr)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Motion detection complete!!")