import pandas as pd
import numpy as np
from PIL import Image
import csv
import glob
import cv2
from matplotlib import pyplot as plt


def grabcut(cv2image, x1, y1, x2, y2):
    mask = np.zeros(cv2image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rectangle = (x1,y1,x2,y2)
   
    cv2.grabCut(cv2image,mask,rectangle,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = cv2image*mask2[:,:,np.newaxis]

    return image

def PIL_to_CV2(PilImage):
    return cv2.cvtColor(np.asarray(PilImage),cv2.COLOR_RGB2BGR) 

def CV2_to_PIL(cv2Image):
    img = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def preprocess(PilImage, x1, y1, x2, y2):
    cv2Image = PIL_to_CV2(PilImage)
   
    cv2Image = cv2.blur(cv2Image, (2,2))
    cv2Image = grabcut(cv2Image, x1, y1, x2, y2)
    cv2Image = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
    cv2Image = cv2.equalizeHist(cv2Image)

    plt.imshow(cv2Image)
    plt.show()

    pilImage = CV2_to_PIL(cv2Image)
    return pilImage

#CONFIG:
LOAD_AND_SHRINK_TRAIN_IMAGES = True


SIZE_IMAGE = 256
CARS = []

with open('names.csv', newline='') as f:
    reader = csv.reader(f)
    CARS = list(reader)


def label_to_vector(index):
    vector = np.zeros(len(CARS))
    vector[index-1] = 1.0
    return vector

data = pd.read_csv('anno_train.csv')
labels_train = []
images_train = []
index = 1
total = data.shape[0]

train_images_files_paths = []
for filename in glob.iglob("archive\\car_data\\car_data\\train" + '**/**', recursive=True):
    train_images_files_paths.append(filename)

def string_containing_substring(train_images_files_paths, substring):
    for s in train_images_files_paths:
        if substring in s:
            return s

if LOAD_AND_SHRINK_TRAIN_IMAGES:
    for index, row in data.iterrows():
        label = label_to_vector(row['label'])
        x1 = (row['value1'])
        y1 = (row['value2'])
        x2 = (row['value3'])
        y2 = (row['value4'])

        image = Image.open(string_containing_substring(train_images_files_paths, row['image']))
        if image is not None:
                labels_train.append(label)
                tmp = preprocess(image, x1, y1, x2, y2)
                tmp.resize((256, 256))
                images_train.append(tmp)
        else:
            print("Error")
        index += 1
        print("Progress train: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    print("Total: " + str(len(labels_train)))
    np.save('files/labels_train.npy', labels_train)
    np.save('files/images_train.npy', images_train)

data = pd.read_csv('anno_test.csv')
labels_test = []
images_test = []
index = 1
total = data.shape[0]

for index, row in data.iterrows():
    label = label_to_vector(row['label'])
    image = Image.open(row['image']).resize((256, 256))
    if image is not None:
            labels_test.append(label)
            images_test.append(image)
    else:
        print("Error")
    index += 1
    print("Progress test: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(images_test)))
np.save('./files/labels_test.npy', labels_test)
np.save('./files/images_test.npy', images_test)