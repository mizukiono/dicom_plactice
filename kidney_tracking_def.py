
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import sys
import datetime
import pydicom
import time


# In[ ]:


def tracking_MEDIANFLOW(videofile, fps, save = False):
    # trackerのしくみを決める
    tracker_type = 'MEDIANFLOW'
    tracker = cv2.TrackerMedianFlow_create()
    
    # Read Video(sizeは512x512を推奨)
    video = cv2.VideoCapture(videofile)
    
    now = datetime.datetime.now()
    timestamp = '{0:%Y%m%d_%H%M%S}'.format(now)
    
    # Output setting
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if save == True:
        out = cv2.VideoWriter('tracking_output_%s.mp4'%timestamp, fourcc , fps, (512,512))
    
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
        
    # Read first frame
    ok, frame = video.read()
    if not ok:
        print("Cannot read the video file")
        sys.exit()
    
    # Define an initial bouding box
    bbox = (144, 295, 60, 40)
        
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    

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
        if save == True:
            out.write(frame)

        #Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    
    # When everything done, release the capture
    video.release()
    if save == True:
        out.release()
    cv2.destroyAllWindows()


# In[ ]:


#　絶対パスで指定したらうまくいった……
tracking_MEDIANFLOW('D:/J-comet/Radiology/dicom/ct_track_code/dicom_images_for_test/jpgs/img_%03d.jpg' , fps = 2.0, save = True)


# In[ ]:


video_input = cv2.VideoCapture('/dicom_image_for_test/jpgs/img_%03d.jpg')
video_input.isOpened()

