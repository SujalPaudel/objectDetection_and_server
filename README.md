# Technical Documentation

With Installation of Anaconda distribution Python, and the python libraries like Pandas, matplotlib, numpy came inbuilt with the distribution. For this object detection the machine learning library Tensorflow is used, and the development environment Jupyter Notebook as it enables us to break down the heavier script and compile it. The final piece of code is written is Linux Environment. Some of my favourite object detection models are SSD[1], Fast RCNN[2], Faster RCNN[3], YOLO[4]. In this project I have used SSD, I have explained it in the article, why have I actually done so.

## Protobuf:
Generally JSON and Protobuf can be used for transferring data between services or system. JSON is comparatively famous than protobuf, as it is more readable, self-contained and extensible, nevertheless it isn’t perfect as in the case of heavy software like Tensorflow, as it happens to be expensive, and when the serialization and deserialization is at high volume(in the case of Tensorflow), the cost happens to be non-negligible. Protobuf is designed by Google.

## Protobuf Compilation:
The tensorflow Object Detection API, uses Protobufs to configure(setup) model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled.
P.S: This compilation changes the proto file to python file

COCO stands for Common Objects in Context

## Tensorflow detection model zoo
For object detection, there exists a number of models, T/F detection model zoo, provides a collection of detection models those are trained on the COCO dataset, Kitti dataset, and Open Image dataset. 

These models are very useful for the out of the box detection/conclusion, if we are based on categorizing the object, it is also supported, as categories already exists in these datasets.

These models can also be used when training on a novel datasets.

## Tensorflow object detection API
The Tensorflow object detection API provides us with the number of models trained on COCO dataset. The models are trained on SSD/ Faster RCNN/ Mask RCNN inception.
So, how should we actually choose the object detection model? Well, it depends primarily in the following factors:
It depends upon our system to choose the model (H/W support, S/W specifications)
Accurate prediction in images can be done with the models having higher speed and higher MAP point.

The primary reason for choosing an SSD(Single Shot MultiBox Detection) is it’s better speed and accuracy in the limited amount of resources.The system uses the TkAgg, with Agg rendering

## Tensorflow works in Graph Principle:- The data flow graph
[1]SSD: SSD is a unified framework for object detection with a single network
[2]Fast RCNN: Fast Regional Convolution Neural Network
[3]Faster RCNN: Faster Regional Convolution Neural Network
[4]YOLO: You Only Look Once

## Efficiency
The code was run for 5 times taking images of child, cat, jewelry, house and car. The system was able to detect the objects with perfection.

Note: I explored many other ways to perform these actions as well, nevertheless I used the tensorflow library, as this will make me familiar with the system and orientation of this library. With some tweaks in the code, we will also be able to figure out the type of the object.

## Server Side
For server side programming python’s flask library is used. It provides a minimalistic MVC pattern support.
