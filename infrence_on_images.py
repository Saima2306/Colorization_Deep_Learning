import cv2 
import glob
from numpy import random
from Model import Net
import torch
import numpy as np
from torch.autograd import Variable
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# Function to extract frames

def to_rgb(grayscale_input, ab_input):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  # plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input.cpu()), 1).detach().numpy() # combine channels
  color_image = color_image.reshape((3,color_image.shape[2],color_image.shape[3]))
  # print(color_image.shape)
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
    img_original = img_original.reshape((1,1,img_original.shape[0],img_original.shape[1]))
    img_original = torch.from_numpy(img_original).float()
    output_ab = net.forward(img_original.cuda())
    color_img = to_rgb(img_original, output_ab)
    color_img = (color_img*255).astype(np.uint8)
    # testx = cv2.cvtColor(testx, cv2.COLOR_RGB2BGR)
    return color_img


  
# Driver Code 
if __name__ == '__main__': 

    image_names = glob.glob('./images/test/*.jpg')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU is available!')
    net = Net()
    if use_gpu:
        net = net.cuda()
    net.load_state_dict(torch.load("/weights/best_detector.pth"))

    for image_name in image_names:
        print(image_name.split('/')[-1])
        img = cv2.imread(image_name)

        col_img = Prediction(img,net)
        cv2.imwrite('output/'+image_name.split('/')[-1],col_img)
  
    # Calling the function 
    # FrameCapture("/media/shahzad/D/Datasets/videoplayback6.mp4") 
