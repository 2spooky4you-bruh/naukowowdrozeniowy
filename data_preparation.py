import pandas as pd
import numpy as np
from PIL import Image
import csv
import glob

#CONFIG:
LOAD_AND_SHRINK_TRAIN_IMAGES = False


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
for filename in glob.iglob("archive" + '**/**', recursive=True):
    train_images_files_paths.append(filename)

def string_containing_substring(train_images_files_paths, substring):
    for s in train_images_files_paths:
        if substring in s:
            return s

if LOAD_AND_SHRINK_TRAIN_IMAGES:
    for index, row in data.iterrows():
        label = label_to_vector(row['label'])
        image = Image.open(string_containing_substring(train_images_files_paths, row['image'])).resize((256, 256))
        if image is not None:
                labels_train.append(label)
                #images_train.append(image)
                image.save("files/" + row['image'])
        else:
            print("Error")
        index += 1
        print("Progress train: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    print("Total: " + str(len(labels_train)))
    np.save('files/labels_train.npy', labels_train)


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