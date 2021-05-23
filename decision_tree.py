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

SHOW_IMAGES = False

def extract_features(image):

    if SHOW_IMAGES:
        imshow(image)
        plt.show()

    image = rgb2gray(image)
    threshold = filters.threshold_mean(image)
    mask = image > threshold
    label_img = measure.label(mask)

    if SHOW_IMAGES:
        imshow(label_img)
        plt.show()

    props = regionprops_table(label_img)

    table = pd.DataFrame(props)
    main_object_features = table.iloc[0]
    return main_object_features

train_images = np.load("files/images_train.npy", allow_pickle=True)
train_labels = np.load("files/labels_train.npy", allow_pickle=True)

train_image_features = []
for image in train_images:
    features = extract_features(image)
    train_image_features.append(features)

test_images = np.load("files/images_test.npy", allow_pickle=True)
test_labels = np.load("files/labels_test.npy", allow_pickle=True)

test_image_features = []
for image in test_images:
    features = extract_features(image)
    test_image_features.append(features)


clf = DecisionTreeClassifier()
clf = clf.fit(train_image_features, train_labels)

predictions = clf.predict(test_image_features)

predictions_argmax=np.argmax(predictions, axis=1)
test_labels_argmax = np.argmax(test_labels, axis=1)

print("Accuracy:", metrics.accuracy_score(test_labels_argmax, predictions_argmax))

#tree.plot_tree(clf)
#plt.show()

#df = pd.DataFrame()

#extract_features(image)