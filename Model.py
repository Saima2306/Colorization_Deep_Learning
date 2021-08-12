
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Autoencoder:
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 8, 4,padding=2,stride=1)
        self.conv2 = nn.Conv2d(8, 16, 4,padding=2,stride=1)
        self.conv3 = nn.Conv2d(16, 32, 4,padding=2,stride=1)
        self.conv4 = nn.Conv2d(32, 64, 4,padding=2,stride=1)
        self.conv5 = nn.Conv2d(64, 128, 4,padding=2,stride=1)
        self.conv6 = nn.Conv2d(128, 256, 4,padding=2,stride=1)
        
        
        self.upsample1 = torch.nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        self.upsample2 = torch.nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        self.upsample3 = torch.nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        self.upsample4 = torch.nn.ConvTranspose2d(32, 16, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        self.upsample5 = torch.nn.ConvTranspose2d(16, 8, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        self.upsample6 = torch.nn.ConvTranspose2d(8, 2, 4, stride = 2, padding = 1, output_padding= 0, dilation = 1)
        
        
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(64 * 35* 35, 120)  # 6*6 from image dimension 
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = F.relu(self.upsample3(x))
        x = F.relu(self.upsample4(x))
        x = F.relu(self.upsample5(x))
        x = F.relu(self.upsample6(x))
        return x



class ColorizationNet_res_18(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet_res_18, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(pretrained=True) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output



class ColorizationNet_res_34(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet_res_34, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet34(pretrained=True) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:-4])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output

class ColorizationNet_res_50(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet_res_50, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet50(pretrained=True) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:-4])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output