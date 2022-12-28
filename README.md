# Classification of Human Activities and Postural Transitions

Authors: Victor Shih, Orion Peter Fernandes  
For UCI MECPS program's ECPS 205 Sensors, Actuators and Sensor Networks course.  

## Table of Contents
* [Quick Overview of File Structures](#Quick-Overview-of-File-Structures)
* [](#)

## Quick Overview of File Structures

Place data files in directories like so. data dir is on the same level as the .py files. 

data/test/X_test.txt  
data/test/y_test.txt  
data/train/X_train.txt  
data/train/y_train.txt  


Training data: 561 features, 7767 samples  
x_train shape = (7767, 561, 1)  
y_train shape = (7767, 1)  
x_test shape = (3162, 561, 1)  
y_test shape = (3162, 1)  

Run mainFile.py to train.  

Results are stored in model_logs.  

## Data source
http://archive.ics.uci.edu/ml/datasets/smartphone-based+recognition+of+human+activities+and+postural+transitions
