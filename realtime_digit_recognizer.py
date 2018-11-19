# import packages
import numpy as np
import cv2
import imutils
from collections import deque
from keras.models import load_model

# Load model
model = load_model('mnist_keras_cnn_model.h5')

# Define upper and lower boundaries of color to be considered
# RED
redLower = (161,134,100)
redUpper = (227,255,255)

kernel = np.ones((5,5),np.uint8)
roi = np.zeros((200, 200, 3), dtype=np.uint8)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
pts = deque(maxlen=512)
ans = ' '

cam = cv2.VideoCapture(0)

while True:
	# grab the current paintWindow
	(grabbed, frame) = cam.read()

	# resize the frame
	frame = imutils.resize(frame, width = 600)
	frame = cv2.flip(frame, 1)

	# blur it to reduce high frequency noise and 
    # allow us to focus on the structural objects inside the frame
	blurred = cv2.GaussianBlur(frame, (11,11), 0)
    
    # convert it to HSV color space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color RED
	mask = cv2.inRange(hsv, redLower, redUpper)
    
    # perform a series of erosions and dilations to remove any
    # small blobs left in the mask
	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.dilate(mask, kernel, iterations=2)
    
    # find contours in the mask and 
    #initialize the current (x,y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	center = None
    
    # Check to see if any contours were found
	if len(cnts)>0:
        # find the largest contour in the mask
		c = max(cnts, key=cv2.contourArea)
        # compute the min enclosing circe and centroid
		((x,y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # proceed if radius meets a minimum size
		if radius > 10:
            # draw circle and centroid on the frame
			cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0,255,255), 2)
			cv2.circle(frame, center, 5, (0,0,255), -1)
            
    	# update the points queue
		pts.appendleft(center)

	elif len(cnts) == 0:
		if len(pts) != 0:
			blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
			blur = cv2.medianBlur(blackboard_gray, 15)
			blur = cv2.GaussianBlur(blur, (5,5), 0)
			thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			b_cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
			
			if len(b_cnts) >0:
				
				c2 = max(b_cnts, key=cv2.contourArea)
				if cv2.contourArea(c2) > 1000:
					x, y, w, h = cv2.boundingRect(c2)
    				# Make the rectangular region around the digit
					roi = blackboard_gray[y-10:y+h+10, x-10:x+w+10]
				    # Resize the image
					roi = cv2.resize(roi, (28, 28))
					roi = cv2.dilate(roi, (3, 3))
					roi = roi.astype('float32')/255
				    # convert to image array
					roi_arr = np.reshape(roi, [1,28,28,1])
				    # predict
					ans = model.predict(roi_arr)[0]
					ans = str(np.argmax(ans))
					print(ans)
			# Empty the points deque
			pts = deque(maxlen=512)
			blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
			cv2.imshow('img',thresh)
			cv2.waitKey()			
				  
	# loop over the set of tracked points
	for i in range(1, len(pts)):
        
        # if the ball was not successfully detected in that given frame
        # ignore the current index continue looping over the pts
		if pts[i-1] is None or pts[i] is None:
			continue
            
        # compute the thickness of the line
		thickness = int(np.sqrt(64 / float(i+1))*2.5)
        # draw the connecting lines
		cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
		cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), thickness)
	cv2.putText(frame, "Result:  "+ans, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
    
	if key == ord("q"):
		break
        
cam.release()
cv2.destroyAllWindows()