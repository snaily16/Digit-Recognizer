# Digit-Recognizer
[![](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/snaily16/Digit-Recognizer/blob/master/LICENSE)

## Description
A deep learning application to recognize digits from real-time handwritten numbers using object tracking and from images.

The Convolutional Neural Network model for this is trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend) which is very intuitive.

## Dependencies
* Python3
* OpenCV 3x
* Keras
* numpy
* imutils
Use pip to install any missing dependencies.

## Usage
To train the model run -

``` 
python3 train.py
```
This will take some time to train the model

To detect digits from image run -

```
python3 image.py
```


For realtime detection run -

```
python3 realtime.py
``` 

## Demo
![Demo](https://github.com/snaily16/Digit-Recognizer/blob/master/numbers.gif)
