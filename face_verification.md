# FaceNet_Recognition

refer to https://github.com/serengil/tensorflow-101/blob/master/python/ArcFace.ipynb 


```python
import cv2
import numpy as np   
#!pip install keras tensorflow==2.4.1 mtcnn Pillow numpy opencv-python matplotlib sklearn
from keras.models import load_model
#!pip install deepface
from deepface.commons import functions, distance as dst
import matplotlib.pyplot as plt
from PIL import Image
import os
```

## Build Facenet Model

Downloaded FaceNet.py from the reference. (same dir)
<br> Used it with a slight modification.


```python
#ref: https://github.com/serengil/deepface/blob/master/deepface/basemodels/FaceNet.py 
import facenet
#ref: https://drive.google.com/uc?id=1971Xk5RwedbudGgTIrGAL4F7Aifu7id1
model=facenet.loadModel()
```

    facenet_weights.h5 will be downloaded...
    

    Downloading...
    From: https://drive.google.com/uc?id=1971Xk5RwedbudGgTIrGAL4F7Aifu7id1
    To: C:\Users\PC921\.deepface\weights\facenet_weights.h5
    92.2MB [00:02, 30.9MB/s]
    

## Import image

** 닮은 꼴로 유명한 김고은, 박소담 으로 test


```python
img1_path = "image1.jpg"
img2_path = "image2.jpg"
img3_path = "image3.jpg"
img4_path = "image4.jpg"
img5_path = "image5.jpg"
img6_path = "image6.jpg"
```

김고은


```python
fig = plt.figure(figsize = (20, 20))

ax1 = fig.add_subplot(1,3,1)
plt.axis('off')
plt.imshow(Image.open(img1_path))

ax1 = fig.add_subplot(1,3,2)
plt.axis('off')
plt.imshow(Image.open(img2_path))

ax1 = fig.add_subplot(1,3,3)
plt.axis('off')
plt.imshow(Image.open(img3_path))

```




    <matplotlib.image.AxesImage at 0x2452bb2f8b0>




    
![output_10_1](https://user-images.githubusercontent.com/80514503/116943414-83c5d700-acae-11eb-8790-cb346ef4b117.png)

    


박소담


```python
fig = plt.figure(figsize = (20, 20))

ax1 = fig.add_subplot(1,3,1)
plt.axis('off')
plt.imshow(Image.open(img4_path))

ax1 = fig.add_subplot(1,3,2)
plt.axis('off')
plt.imshow(Image.open(img5_path))

ax1 = fig.add_subplot(1,3,3)
plt.axis('off')
plt.imshow(Image.open(img6_path))
```




    <matplotlib.image.AxesImage at 0x245465847f0>




    
![png](output_12_1.png)![output_12_1](https://user-images.githubusercontent.com/80514503/116943433-8d4f3f00-acae-11eb-9356-1126b2a2124e.png)

    


## Detect, Align

Facenet input size = 160 x 160 <br>
detector_backend는 opencv 사용할 예정


```python
#ref: https://github.com/serengil/deepface
import detect_align
```


```python
detector_backend = 'opencv'

detect_align.initialize_detector(detector_backend)
#detect and align 

img1 = detect_align.preprocess_face(img1_path, target_size = (160, 160), detector_backend = detector_backend)
img2 = detect_align.preprocess_face(img2_path, target_size = (160, 160), detector_backend = detector_backend)
img3 = detect_align.preprocess_face(img3_path, target_size = (160, 160), detector_backend = detector_backend)
img4 = detect_align.preprocess_face(img4_path, target_size = (160, 160), detector_backend = detector_backend)
img5 = detect_align.preprocess_face(img5_path, target_size = (160, 160), detector_backend = detector_backend)
img6 = detect_align.preprocess_face(img6_path, target_size = (160, 160), detector_backend = detector_backend)

#find vector embeddings
img1_embedding = model.predict(img1)
img2_embedding = model.predict(img2)
img3_embedding = model.predict(img3)
img4_embedding = model.predict(img4)
img5_embedding = model.predict(img5)
img6_embedding = model.predict(img6)
```

## Represent, Verify

#### Distance function

cosine = 0.40 ,  euclidean = 10 , euclidean_l2 = 0.80 <br>
Use cosine 사용


```python
def CosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def EuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))
```


```python
def verify(img1, img2):
    
    #representation
    
    img1_embedding = model.predict(img1)[0]
    img2_embedding = model.predict(img2)[0]
    
    distance = CosineDistance(img1_embedding, img2_embedding)
    
    #------------------------------
    #display
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img1[0][:,:,::-1])
    
    ax2 = fig.add_subplot(1,2,2)
    plt.axis('off')
    plt.imshow(img2[0][:,:,::-1])
    
    plt.show()
    
    #------------------------------
    #verification
    
    threshold = findThreshold(metric)
    
    if distance <= 0.40:
        print("True")
    else: 
        print("False")
    
    print("Distance = ",round(distance, 2))
    
    
```

## True Image


```python
#------------------------------
#김고은
verify(img1, img2)
verify(img2, img3)
verify(img1, img3)

#------------------------------
#박소담
verify(img4, img5)
verify(img5, img6)
verify(img4, img6)
```


    
![png](output_23_0.png)
    


    True
    Distance =  0.18
    


    
![png](output_23_2.png)
    


    True
    Distance =  0.12
    


    
![png](output_23_4.png)
    


    True
    Distance =  0.17
    


    
![png](output_23_6.png)
    


    True
    Distance =  0.14
    


    
![png](output_23_8.png)
    


    True
    Distance =  0.17
    


    
![png](output_23_10.png)
    


    True
    Distance =  0.19
    

## False Image


```python
verify(img1, img4)
verify(img2, img5)
verify(img3, img6)
verify(img1, img5)
verify(img2, img6)
verify(img3, img4)
verify(img1, img6)
verify(img2, img4)
verify(img3, img5)
```


    
![png](output_25_0.png)
    


    False
    Distance =  0.77
    


    
![png](output_25_2.png)
    


    False
    Distance =  0.59
    


    
![png](output_25_4.png)
    


    False
    Distance =  0.57
    


    
![png](output_25_6.png)
    


    False
    Distance =  0.65
    


    
![png](output_25_8.png)
    


    False
    Distance =  0.5
    


    
![png](output_25_10.png)
    


    False
    Distance =  0.73
    


    
![png](output_25_12.png)
    


    False
    Distance =  0.59
    


    
![png](output_25_14.png)
    


    False
    Distance =  0.69
    


    
![png](output_25_16.png)
    


    False
    Distance =  0.62
    

## 고려사항 

1. 존재하는 가중치들을 사용할 것인지, data를 구해서 학습을 다시 시킬 것인지 ***
2. openCV를 backend detector로 사용할지 다른 backend detector를 사용할지 고민

## 계획

1. database에 임베딩 저장 구현
2. web cam과 face net 연결
3. 여러 비교 대상이 왔을때 어떻게 들고 와서 결과를 낼 지에 대한 방법 모색......
