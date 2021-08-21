# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:21:36 2021

@author: kiran gupta
"""

import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from model import label_encoder, model
import random as rng
rng.seed(12345)

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def get_coordinates(contours):
    all_points=[]
    for i in range(len(contours)):
        contour = contours[i]
        temp = (contour)
        #print(temp)
        x_list=[]
        y_list=[]
        for j in range(len(temp)):
            x_list.append(temp[j][0][0])
            y_list.append(temp[j][0][1])
        
        x1 = min(x_list)
        y1 = min(y_list)
        x2 = max(x_list)
        y2 = max(y_list)
        all_points.append((x1,y1,x2,y2))
    return all_points
        
def crop_and_save_contours(all_points,filename):
    filename = 'demo.png'
    # load image from file
    im=Image.open(filename)
    path=[]
    count=0
    for points in all_points:  

        x1 = points[0]-5
        y1= points[1]-5
        x2 = points[2]+5
        y2= points[3]+5
    
        im1 = im.crop((x1,y1,x2,y2))
        image_name = str(count)+".jpg"
        im1.save("test_images/"+image_name)
        path.append("test_images/"+image_name)
        count=count+1
    return path

def pred(path,new_model):
    predictions=[]
    for p in path[:100]:  
        image = cv.imread(p)
        output = cv.resize(image, (32,32))
        
        output = output.reshape((1,32,32,3))
        
        result = new_model.predict(output)
        result1 = result.tolist()
        pred = result1[0].index(max(result1[0]))
        predictions.append(pred)
        
    return predictions

def get_text_from_pred(label_encoder,predictions):
    predictions_test = label_encoder.inverse_transform(predictions)
    return predictions_test
    

if __name__=="__main__":
    # Load source image
    src = cv.imread("demo.png")
    new_model = model
    
    print(src.shape)
    if src is None:
        print('Could not open or find the image:')
        exit(0)
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))
    
    threshold = 100
    #Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    #Find contours
    _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contours.sort(key=lambda x:get_contour_precedence(x, src.shape[1]))
    
    all_points = get_coordinates(contours)
    paths=crop_and_save_contours(all_points,src)
    predictions = pred(paths,new_model)
    
    output_text = get_text_from_pred(label_encoder,predictions)
    #Write to a Text file
    my_file = open("NewFile.txt", "w+")
    my_file.write(output_text)
    my_file.close()
    
    