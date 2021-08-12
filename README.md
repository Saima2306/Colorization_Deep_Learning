# Colorization_Deep_Learning
This respository is used for Colorization of black and white images using Lab color space on videos

# Requirements:
```
Python == 3.6
pytorch >= 0.4.0 or 1.0
```
# Getting Started:
### Clone this repository:
```
  git clone https://github.com//Saima2306/Colorization.git
  cd Colorization
 ```
 ### Frame extraction from Videos:
 
 Run: ```python frame_extraction.py```

Frames will be extracted into the folder,images 

### Training:
``` python train.py ```
### Testing:
#### Testing on Images:
```python inferenece_on_images.py```<br />

Results will be saved in outputs

#### Testing on Videos:
```python inference_on_videos.py```<br />

Result will be saved in out_video
