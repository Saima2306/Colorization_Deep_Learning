import cv2 
import glob
from numpy import random
# Function to extract frames 
count_train = 0
count_test = 0

def FrameCapture(path): 
      
    # Path to video file 
    global count_test,count_train 
    vidObj = cv2.VideoCapture(path) 
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()

        if success:
            x = random.rand()
            if x > 0.98:
                image = cv2.resize(image,(512,256))
                y = random.rand()
                if y>0.1:
                    cv2.imwrite("Image/train/frame%d.jpg" % count_train, image)
                    count_train += 1
                else:
                    cv2.imwrite("Image/test/frame%d.jpg" % count_test, image)
                    count_test += 1

  
# Driver Code 
if __name__ == '__main__': 

    video_names = glob.glob('./data/*.webm')

    for video_name in video_names:
        print(video_name.split('/')[-1])
        FrameCapture(video_name)