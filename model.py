# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 19:20:15 2021

@author: kiran
"""
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

def build_model():
    #Build a model
    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(32,32,3), pooling=None, classes=27
    )
    gap = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
    fc1 = tf.keras.layers.Dense(27, activation='relu')(gap)
    model = tf.keras.models.Model(inputs=resnet.input, outputs=fc1)
    return model

def create_dataset():
    x_train=[]
    y_train=[]
    for i in range(len(dataset)):
        path = "test_images/"+str(dataset[i][0])+".jpg"
        img = cv.imread(path)
        img = cv.resize(img,(32,32))
        x_train.append(img)
        y_train.append(dataset[i][1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train


if __name__=="__main__":
    dataset = pd.read_csv("y_train_ocr.csv")
    dataset = dataset.values

    x_train,y_train = create_dataset()
    label_encoder = preprocessing.LabelEncoder()
    y_data= label_encoder.fit_transform(y_train) 
    
    model = build_model()
    model.compile(optimizer="Adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(x_train,y_data,epochs=20)
    
    #Evaluate model
    model.evaluate(x_train,y_data)