
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import imresize, imread, imshow
import scipy
import cv2
import time
import os
from Model import Net

class GrayscaleImageFolder(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab, target

# Training
train_transforms = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip()])
train_imagefolder = GrayscaleImageFolder('images/train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=128, shuffle=True)

# Validation 
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])
val_imagefolder = GrayscaleImageFolder('images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=128, shuffle=False)

# Check availability of GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available!')

net = Net()
print(net)
if use_gpu:
    net = net.cuda()


# ## Define loss function and optimizer

criterion = nn.L1Loss() # 2D Negative Log-Likelihood
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-2)


# ## Train the network

iterations = 5
trainLoss = []
testLoss = []
start = time.time()
best_test_loss = np.inf
for epoch in range(iterations):
    epochStart = time.time() 
    runningLoss = 0   
    net.train(True) # For training
    for i,(input_gray, input_ab, target) in enumerate(train_loader):
        if use_gpu:
            input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()      
 
        
        # Feed-forward input data through the network
        output_ab = net(input_gray)
        # Compute loss/error
        loss = criterion(output_ab, input_ab)      
        # Initialize gradients to zero
        optimizer.zero_grad()                  
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.item()
        if i % 25 == 0:
          print('epoch:[{0}]/[{1}/{2}]\t''loss = {3}'.format(epoch, i, len(train_loader), loss.item()))          
    avgTrainLoss = runningLoss/len(train_loader)   
    trainLoss.append(avgTrainLoss)
  
    
    # Evaluating performance on test set for each epoch
    net.train(False) # For testing
    test_runningLoss = 0    
    for input_gray, input_ab, target in val_loader:
        # Wrap them in Variable
        if use_gpu:
            input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()
        # else:
        #     inputs, labels = Variable(inputs), Variable(labels)         
        output_ab = net(input_gray)       
         # Compute loss/error
        loss = criterion(output_ab, input_ab)      
        # Accumulate loss per batch
        test_runningLoss += loss.item() 
        
    avgTestLoss = test_runningLoss/len(val_loader)    
    testLoss.append(avgTestLoss)
        
    # Plotting Loss vs Epochs
    fig1 = plt.figure(1)        
    plt.plot(range(epoch+1),trainLoss,'r--',label='train')        
    plt.plot(range(epoch+1),testLoss,'g--',label='test')        
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')   
      
    
    epochEnd = time.time()-epochStart
    print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f}; Time consumed: {:.0f}m {:.0f}s '.format(epoch + 1,iterations,avgTrainLoss,epochEnd//60,epochEnd%60))
    print('At Iteration: {:.0f} /{:.0f}  ;  Testing Loss: {:.6f} ; Time consumed: {:.0f}m {:.0f}s '.format(epoch + 1,iterations,avgTestLoss,epochEnd//60,epochEnd%60))
    
    if best_test_loss > avgTestLoss:
        best_test_loss = avgTestLoss
        print('Updating best test loss: %.5f' % best_test_loss)
        torch.save(net.state_dict(),'/weights/best_detector.pth')
    torch.save(net.state_dict(),'/weights/bestweight.pth')
    
    
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
torch.save(net.state_dict(),"/weights/bestweight.pth"%(epoch+1))







