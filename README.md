# DozeNet
Simple and Fast neural network for sleep apnea detection

run 
``` python MobileNetV3.py ``` 
after using the required function from the MobileNetV3 python file which includes:
- ```CNN+ELU``` under ```define_model()```
- ```MobileNetV3Small+H_Swish```
- ```MobileNetV3Small+H_Swish```
- ```MobileNetV3Large+GELU```
- ```MobileNetV3Large+GELU```

The above requires you to have ```X.npy``` and ```Y.py``` which are the training data after fetching the data from MIT-BIH followed by converting ```.st``` files to ```.csv``` files and followed by reading each file and adding them to respective numpy files.

