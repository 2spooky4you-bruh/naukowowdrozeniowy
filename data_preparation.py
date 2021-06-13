import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import csv
import glob
import cv2
from matplotlib import pyplot as plt
import random
import imutils

#CONFIG:
LOAD_AND_PREPROCESS_TRAIN_IMAGES = True
LOAD_AND_PREPROCESS_TEST_IMAGES = False
SHOW_IMAGES = False

#SIZE_IMAGE = 128
#ADJUST_GAMMA = 1.0
#GRABCUT = False
#EQUALIZE_HIST = False
#ROTATE = False
#BRIGHTNESS = 1.0
#BLUR = False

def adjust_gamma(image, gamma=1.0):
    #https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def grabcut(cv2image, x1, y1, x2, y2):
    mask = np.zeros(cv2image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rectangle = (x1,y1,x2,y2)
   
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_RGBA2RGB)
    cv2.grabCut(cv2image,mask,rectangle,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = cv2image*mask2[:,:,np.newaxis]

    return image

def PIL_to_CV2(PilImage):
    open_cv_image = np.array(PilImage) 
    #open_cv_image = open_cv_image[:, :, :].copy() 
    return open_cv_image


def CV2_to_PIL(cv2Image):
    #img = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cv2Image)
    return im_pil

def preprocess(PilImage, x1, y1, x2, y2):
    cv2Image = PIL_to_CV2(PilImage)
    not_processed = PIL_to_CV2(PilImage)

    if GRABCUT:
        cv2Image = grabcut(cv2Image, x1, y1, x2, y2) 

    if BLUR:
        cv2Image = cv2.blur(cv2Image, (3,3))

    if EQUALIZE_HIST:
        if len(cv2Image.shape)== 3:
            cv2Image = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
            cv2Image = cv2.equalizeHist(cv2Image)

    if ROTATE:
        cv2Image = imutils.rotate_bound(cv2Image, random.randint(0, 360))

    if ADJUST_GAMMA != 1.0:
        cv2Image  = adjust_gamma(cv2Image , gamma=ADJUST_GAMMA)
    
    if BRIGHTNESS != 1.0:
        pilImage = CV2_to_PIL(cv2Image)
        enhancer = ImageEnhance.Brightness(pilImage)
        pilImage = enhancer.enhance(BRIGHTNESS)
        cv2Image = PIL_to_CV2(pilImage)

    if SHOW_IMAGES:  
        #plt.imshow(np.hstack([not_processed, cv2Image]))
        plt.imshow(cv2Image)
        plt.show()

    pilImage = CV2_to_PIL(cv2Image)
    return pilImage

def process_images():
    CARS = []

    with open('names.csv', newline='') as f:
        reader = csv.reader(f)
        CARS = list(reader)


    def label_to_vector(index):
        vector = np.zeros(len(CARS))
        vector[index-1] = 1.0
        return vector

    config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)

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

    if LOAD_AND_PREPROCESS_TRAIN_IMAGES:
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
                    #tmp = tmp.resize((SIZE_IMAGE, SIZE_IMAGE))
                    tmp = PIL_to_CV2(tmp)
                    if len(tmp.shape)== 3:
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    dim=(SIZE_IMAGE, SIZE_IMAGE)
                    tmp= cv2.resize(tmp, dim, interpolation = cv2.INTER_AREA)
                    #tmp= cv2.blur(tmp, (5,5))
                    tmp = np.array(tmp)
                    images_train.append(tmp)
            else:
                print("Error")
            index += 1
            print("Progress train: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

        print("Total: " + str(len(labels_train)))
        np.save('files/labels_train.npy', labels_train)
        np.save('files/images_train' + config_name + '.npy', images_train)

    data = pd.read_csv('anno_test.csv')
    labels_test = []
    images_test = []
    index = 1
    total = data.shape[0]

    test_images_files_paths = []
    for filename in glob.iglob("archive\\car_data\\car_data\\test" + '**/**', recursive=True):
        test_images_files_paths.append(filename)

    if LOAD_AND_PREPROCESS_TEST_IMAGES:
        for index, row in data.iterrows():
            label = label_to_vector(row['label'])
            x1 = (row['value1'])
            y1 = (row['value2'])
            x2 = (row['value3'])
            y2 = (row['value4'])
            image = Image.open(string_containing_substring(test_images_files_paths, row['image']))
            if image is not None:
                    labels_test.append(label)
                    tmp = preprocess(image, x1, y1, x2, y2)
                    tmp = PIL_to_CV2(tmp)
                    if len(tmp.shape)== 3:
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    dim=(SIZE_IMAGE, SIZE_IMAGE)
                    tmp= cv2.resize(tmp, dim, interpolation = cv2.INTER_AREA)
                    #tmp= cv2.blur(tmp, (5,5))
                    tmp = np.array(tmp)
                    images_test.append(tmp)
            else:
                print("Error")
            index += 1
            print("Progress test: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

        print("Total: " + str(len(images_test)))
        np.save('./files/labels_test.npy', labels_test)
        np.save('./files/images_test' + config_name + '.npy', images_test)


#zapisanie zdjęc:
SIZE_IMAGE = 128 #1
ADJUST_GAMMA = 1.0
GRABCUT = True ###
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
process_images()

SIZE_IMAGE = 128 #2
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #3
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = True ###
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #4
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = True ###
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #5
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #6
ADJUST_GAMMA = 0.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #7
ADJUST_GAMMA = 1.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #8
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 0.5 ###
BLUR = False
#process_images()

SIZE_IMAGE = 128 #10
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.5 ###
BLUR = False
#process_images()

SIZE_IMAGE = 64 #11 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 196 #12 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
#process_images()

SIZE_IMAGE = 128 #13
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = True ###
#process_images()