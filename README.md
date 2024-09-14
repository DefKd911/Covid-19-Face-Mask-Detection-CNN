
#  COVID -19 facemask detection 
This project implements a deep learning model to detect whether a person is wearing a face mask or not, 
using computer vision techniques. It's designed to help enforce mask-wearing policies in public spaces during the COVID-19 pandemic.

# Features
-> Detects presence of face masks in images

->Uses transfer learning with VGG16 for efficient training

->Achieves high accuracy on mask detection task

->Includes tools for model evaluation and visualization

# Model Architecture (VGG-16)
![download](https://github.com/user-attachments/assets/8d5a9483-1ba1-455f-adad-6108f56cce2b)

This project uses transfer learning with the VGG16 model pre-trained on ImageNet. The base VGG16 model is followed by:

->Global Average Pooling

->Dense layer (256 units, ReLU activation)

->Output Dense layer (1 unit, Sigmoid activation)



# Dataset
The model is trained on the Face Mask Detection dataset from Kaggle, which includes:

3725 images of people wearing masks & 
3828 images of people not wearing masks


# Future Improvements

Implement real-time mask detection using webcam feed

Explore other pre-trained models (e.g., MobileNet, ResNet)

Expand dataset to include more diverse images

Implement object detection to locate faces before classification
