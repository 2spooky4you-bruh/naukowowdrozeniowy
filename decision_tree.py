import skimage
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

image = rgb2gray(imread("E:\\1 semestr magisterka\\Projekt wdrozeniowy\\archive\\car_data\\car_data\\train\Acura RL Sedan 2012\\00858.jpg"))
imshow(image)
#plt.show()

binary = image < threshold_otsu(image)
binary = closing(binary)
label_img = label(binary)

imshow(label_img)
plt.show()

df = pd.DataFrame()

table = pd.DataFrame(regionprops_table(label_img, image,
                                       ['convex_area', 'area',
                                        'eccentricity', 'extent',                   
                                        'inertia_tensor',
                                        'major_axis_length', 
                                        'minor_axis_length', 'mean_intensity']))

#table['perimeter_area_ratio'] = table['perimeter']/table['area']
real_images = []
std = []
mean = []
percent25 = []
percent75 = [] 

print(table)

for prop in regionprops(label_img): 
      
    min_row, min_col, max_row, max_col = prop.bbox
    img = image[min_row:max_row,min_col:max_col]
    real_images += [img]
    mean += [np.mean(img)]
    std += [np.std(img)]
    percent25 += [np.percentile(img, 25)] 
    percent75 += [np.percentile(img, 75)]
    table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']
    table['label'] = 'a'
    df = pd.concat([df, table], axis=0)

    print(table)