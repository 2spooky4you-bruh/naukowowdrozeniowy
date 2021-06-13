from pandas.core.frame import DataFrame
import skimage
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skimage import data, filters, measure, morphology
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from os.path import join

SHOW_IMAGES = False


def extract_features(image):

    if SHOW_IMAGES:
        imshow(image)
        plt.show()

    #image = rgb2gray(image)
    threshold = filters.threshold_mean(image)
    mask = image > threshold
    label_img = measure.label(mask)

    if SHOW_IMAGES:
        imshow(label_img)
        plt.show()

    props = regionprops_table(label_img)

    tmp = np.array([0,0,0,0,0])
    table = pd.DataFrame(props)
    if table.size>0:
        main_object_features = table.iloc[0]
        main_object_features = np.array(main_object_features)
        tmp = main_object_features
        return main_object_features
    else:
        return tmp
    
    

#train_images = np.load("files/images_train.npy", allow_pickle=True)
#train_labels = np.load("files/labels_train.npy", allow_pickle=True)
#test_images = np.load("files/images_test.npy", allow_pickle=True)
#test_labels = np.load("files/labels_test.npy", allow_pickle=True)

def decision_tree():

    images_train = np.load(join('./files', 'images_train' + config_name + '.npy'), allow_pickle=True)
    labels_train = np.load(join('./files', 'labels_train.npy'), allow_pickle=True)

    #inputs = np.concatenate((train_images, test_images), axis=0)
    #targets = np.concatenate((train_labels, test_labels), axis=0)

    inputs =  images_train
    targets = labels_train

    image_features = []
    for image in inputs:
        features = extract_features(image)
        image_features.append(features)

    for row in image_features:
        while len(row) < 5:
            row.append(0)

    kfold = KFold(n_splits=5, shuffle=False)

    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        clf = DecisionTreeClassifier(random_state=1)

        train_labels_argmax = np.argmax(targets[train], axis=1)
        clf = clf.fit(np.array(image_features)[train], targets[train])

        accuracy = []
        f1 = []
        mathious = []
        recall = []
        precision = []

        test_labels_argmax = np.argmax(targets[test], axis=1)
        predictions = clf.predict(np.array(image_features)[test])

        predictions_argmax=np.argmax(predictions, axis=1)
        #test_labels_argmax = np.argmax(targets[test], axis=1)

        print("Accuracy:", metrics.accuracy_score(test_labels_argmax, predictions_argmax))
        print("F1 Score:", metrics.f1_score(test_labels_argmax, predictions_argmax, average='weighted'))
        print("Mathious correlacion coefficient:", metrics.matthews_corrcoef(test_labels_argmax, predictions_argmax))
        print("Recall:", metrics.recall_score(test_labels_argmax, predictions_argmax, average='weighted'))
        print("Precision:", metrics.precision_score(test_labels_argmax, predictions_argmax, average='weighted'))


        accuracy.append(metrics.accuracy_score(test_labels_argmax, predictions_argmax))
        f1.append(metrics.f1_score(test_labels_argmax, predictions_argmax, average='weighted'))
        mathious.append(metrics.matthews_corrcoef(test_labels_argmax, predictions_argmax))
        recall.append(metrics.recall_score(test_labels_argmax, predictions_argmax, average='weighted'))
        precision.append(metrics.precision_score(test_labels_argmax, predictions_argmax, average='weighted'))

        df = pd.DataFrame({"Config" : config_name, "Fold" : fold_no, "Accuracy" : accuracy, "F1 Score" : f1, "Mathious" : mathious, "Recall" : recall, "Precision" : precision})
        with open("DT.csv", 'a') as f:
            df.to_csv(f, header=None)

        fold_no = fold_no + 1

SIZE_IMAGE = 128 #1
ADJUST_GAMMA = 1.0
GRABCUT = True ###
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #2
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #3
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = True ###
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #4
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = True ###
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()


SIZE_IMAGE = 128 #5
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #6
ADJUST_GAMMA = 0.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #7
ADJUST_GAMMA = 1.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #8
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 0.5 ###
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #9
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.5 ###
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 64 #10 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 196 #11 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

SIZE_IMAGE = 128 #12
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = True ###
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
decision_tree()

#tree.plot_tree(clf)
#plt.show()

#df = pd.DataFrame()

#extract_features(image)