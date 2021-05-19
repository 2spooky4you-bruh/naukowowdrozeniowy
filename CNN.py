import tflearn as tfl
import numpy as np
from os.path import join
import os

SIZE_IMAGE = 256
CARS = ['', '', '', '', '', '', '']


class Recognition:

    def build_network(self):

        self.network = tfl.input_data(shape=[None, SIZE_IMAGE, SIZE_IMAGE, 1])
        self.network = tfl.conv_2d(self.network, 64, 5, activation='relu')
        self.network = tfl.max_pool_2d(self.network, 3, strides=2)
        self.network = tfl.conv_2d(self.network, 64, 5, activation='relu')
        self.network = tfl.max_pool_2d(self.network, 3, strides=2)
        self.network = tfl.conv_2d(self.network, 128, 4, activation='relu')
        self.network = tfl.dropout(self.network, 0.3)
        self.network = tfl.fully_connected(self.network, 3072, activation='relu')
        self.network = tfl.fully_connected(
        self.network, len(CARS), activation='softmax')

        self.network = tfl.regression(
            self.network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )

        self.model = tfl.DNN(
            self.network,
            checkpoint_path='./files',
            max_checkpoints=1,
            tensorboard_verbose=2
        )

    def train_net(self):
        _images_train = np.load(join('./files', 'images_train.npy'))
        self.images_train = _images_train.reshape([-1, SIZE_IMAGE, SIZE_IMAGE, 1])
        _labels_train = np.load(join('./files', 'labels_train.npy'))
        self.labels_train = _labels_train.reshape([-1, len(CARS)])

        _images_test = np.load(join('./files', 'images_test.npy'))
        self.images_test = _images_test.reshape([-1, SIZE_IMAGE, SIZE_IMAGE, 1])
        _labels_test = np.load(join('./files', 'labels_test.npy'))
        self.labels_test = _labels_test.reshape([-1, len(CARS)])


        self.build_network()

        self.model.fit(
            self.images_train, self.labels_train,
            validation_set=(self.images_test,
                            self.labels_test),
            n_epoch=20,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )

        self.save_model()

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
