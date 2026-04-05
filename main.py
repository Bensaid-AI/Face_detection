import os
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tqdm import tqdm_notebook as tqdm
TF_ENABLE_ONEDNN_OPTS=0
image_paths = []
image_names = []
image_dir = "Dataset/"
for image_name in tqdm(os.listdir(image_dir)) : 
    image_path = image_dir + image_name
    image_paths.append(image_path)
    image_names.append(image_name)  
image_dataframe = pd.DataFrame(index= np.arange(len(image_names)),columns=["image_name","path"])
i = 0 
for name ,path in tqdm(zip(image_names,image_paths)) : 
    image_dataframe.iloc[i]["image_name"] = name 
    image_dataframe.iloc[i]["path"] = path
    i = i + 1
simple_images = []
def get_images() : 
    sample_images = []
    random_image_paths = [np.random.choice(image_dataframe["path"])for i in range(6)]
    plt.figure(figsize=(12,8))
    for i in range(6) : 
        plt.subplot(2,3,i+1)
        image= cv.imread(random_image_paths[i])
        image = cv.cvtColor(image , cv.COLOR_BGR2RGB)
        sample_images.append(image)
        plt.imshow(image,cmap= "gray")
        plt.grid(False)
    plt.tight_layout() 
    plt.show()
    return sample_images
simple_images = get_images() 

def load_my_image() : 
    my_image_01 = cv.imread("face1.jpg")
    my_image_01 = cv.cvtColor(my_image_01, cv.COLOR_BGR2RGB)

    my_image_02 = cv.imread("face1.jpg")
    my_image_02 = cv.cvtColor(my_image_02, cv.COLOR_BGR2RGB)
    return my_image_01,my_image_02

img1,img2 = load_my_image()

def haar_cascade_detection(simple_images):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for image in tqdm(simple_images) : 
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 5, 50)
        
        for (x_coordinate, y_coordinate, height, width) in faces : 
            cv.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (100, 0, 0), 2)

haar_cascade_detection([img1,img2])
# plt.figure(figsize = (12, 8))
# for i in range(6) : 
#     plt.subplot(2,3, i+1)
#     plt.imshow(simple_images[i], cmap = "gray")
#     plt.title("Image {}".format(i+1))
#     plt.grid(False)
# plt.tight_layout()

plt.figure(figsize = (12, 8))
plt.subplot(1,2,1) 
plt.title("The One During A Conference", fontsize = 13)
plt.imshow(img1, cmap = "gray")
plt.grid(False)

plt.subplot(1,2,2) 
plt.title("The One With Home Buddies!", fontsize = 13)
plt.imshow(img2, cmap = "gray")
plt.grid(False)

plt.tight_layout()
plt.show()






 






