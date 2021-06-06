import tflearn as tfl
import numpy as np
from os.path import join
import os
import csv
from sklearn.model_selection import KFold
from sklearn import metrics
from math import *
from tflearn import init_graph
from skimage.io import imshow, imread
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


SIZE_IMAGE = 128
SHOW_IMAGES = False

with open('names.csv', newline='') as f:
    reader = csv.reader(f)
    CARS = list(reader)


class Recognition:

    def build_network(self):

        self.model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=[SIZE_IMAGE, SIZE_IMAGE, 1]),
            layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.1),
            layers.Conv2D(64, kernel_size=(4, 4), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.1),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.1),
            layers.Conv2D(64, kernel_size=(2, 2), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(len(CARS), activation="softmax"),
        ])
      
        self.model.summary()

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    def train_net(self):
        from tensorflow import set_random_seed
        set_random_seed(1)

        self.images_train = np.load(join('./files', 'images_train.npy'), allow_pickle=True)
        self.labels_train = np.load(join('./files', 'labels_train.npy'), allow_pickle=True)

        self.images_test = np.load(join('./files', 'images_test.npy'), allow_pickle=True)
        self.labels_test = np.load(join('./files', 'labels_test.npy'), allow_pickle=True)

        self.images_train = self.images_train /255.0
        self.images_test = self.images_test /255.0

        self.images_train = self.images_train.reshape(len(self.images_train), SIZE_IMAGE, SIZE_IMAGE, 1)
        self.images_test = self.images_test.reshape(len(self.images_test), SIZE_IMAGE, SIZE_IMAGE, 1)

        self.inputs = np.concatenate((self.images_train, self.images_test), axis=0)
        self.targets = np.concatenate((self.labels_train,  self.labels_test), axis=0)

        if SHOW_IMAGES:
            i=0
            for image in self.inputs:
                imshow(image)
                plt.title(self.targets[i])
                plt.show()
                i = i +1

        self.build_network()

        kfold = KFold(n_splits=2, shuffle=False)

        fold_no = 1
        for train, test in kfold.split(self.inputs, self.targets):

            self.model.fit(self.inputs[train], self.targets[train], validation_data=(self.inputs[test], self.targets[test]), batch_size=64, epochs=40)

            #scores = self.model.evaluate(self.inputs[test], self.targets[test])

            predictions = self.model.predict(self.inputs[test])

            predictions_argmax=np.argmax(predictions, axis=1)
            test_labels_argmax = np.argmax(self.targets[test], axis=1)

            print("Accuracy:", metrics.accuracy_score(test_labels_argmax, predictions_argmax))
            print("F1 Score:", metrics.f1_score(test_labels_argmax, predictions_argmax, average='weighted'))
            print("Mathious correlacion coefficient:", metrics.matthews_corrcoef(test_labels_argmax, predictions_argmax))


            fold_no = fold_no + 1

        #self.save_model()

    def save_model(self):
        self.model.save(join('./files', 'saved_model'))
        print('[+] Model trained and saved at ' + 'saved_model')

    def load_model(self):
        if os.path.isfile(join('./files', 'saved_model')):
            self.model.load(join('./files', 'saved_model'))
            print('[+] Model loaded from ' + 'saved_model')

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([1, SIZE_IMAGE, SIZE_IMAGE, -1])
        return self.model.predict(image)
