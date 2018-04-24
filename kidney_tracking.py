
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import sys
import time


# In[ ]:


# trackerのしくみを決める
tracker_type = 'MEDIANFLOW'
tracker = cv2.TrackerMedianFlow_create()


# In[ ]:


# Read Video
video = cv2.VideoCapture("test_tracking.avi")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('tracking_output.mp4',fourcc , 20.0, (512,512))


# In[ ]:


# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()


# In[ ]:


# Read first frame
ok, frame = video.read()
if not ok:
    print("Cannot read the video file")
    sys.exit()


# In[ ]:


# Define an initial bouding box
bbox = (129, 285, 60, 40)


# In[ ]:


# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


# In[ ]:


while True:
    # read a new frame
    ok, frame = video.read()
    if not ok:
        time.sleep(2)
        break
    
    # start timer
    timer = cv2.getTickCount()
    
    # update tracker
    ok, bbox = tracker.update(frame)
    
    #calculate frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
    # Draw bouding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    
    # Display result
    cv2.imshow("Tracking", frame)
    
    # Save result as movie file
    out.write(frame)
    
    #Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


# In[ ]:


# When everything done, release the capture
video.release()
out.release()
cv2.destroyAllWindows()

