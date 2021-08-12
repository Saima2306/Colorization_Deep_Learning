import numpy as np
import cv2
import glob
from numpy import random
from Model import Net
import torch
import numpy as np
from torch.autograd import Variable

def to_rgb(grayscale_input, ab_input):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  # plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  return color_image 

def Prediction(img,net):
    img_original = np.asarray(img)
    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    output_ab = net.forward(img_original.cuda())
    color_img = to_rgb(img_original, output_ab)
    return color_img


def Video_prediction(video_name,net):
    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out_video/'+video_name.split('/')[-1],fourcc, 20.0, (512,256))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(frame,(512,256))
            frame = Prediction(frame,net)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

  
# Driver Code 
if __name__ == '__main__': 

    video_names = glob.glob('./test_video/*.mp4')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU is available!')
    net = Net()
    if use_gpu:
        net = net.cuda()
    net.load_state_dict(torch.load("/weights/best_detector.pth"))

    for video_name in video_names:
        print(video_name.split('/')[-1])
        Video_prediction(video_name,net)

