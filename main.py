import os
import numpy as np
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
print(image_dataframe.head())



