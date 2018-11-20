# import packages
import numpy as np
import cv2
from keras.models import load_model

model = load_model('mnist_keras_cnn_model.h5')

image_path = 'images/digit.jpg'
im = cv2.imread(image_path)
im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# add padding
image = cv2.copyMakeBorder(im_g, 0,0,1,1,cv2.BORDER_CONSTANT)

im_gray = cv2.GaussianBlur(image, (5,5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# find contours in the image
cnts= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

# get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in cnts]
# loop over the countours
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 2) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    
    # convert to image array
    roi_arr = np.reshape(roi, [1,28,28,1])
    # predict
    c = model.predict(roi_arr)[0]
    op=str(np.argmax(c))
    #print(op)
    cv2.putText(im, op, (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)
# then display the output contours
cv2.imshow('image',im)
cv2.waitKey()
