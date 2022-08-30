import os
import sys
import time
import math
from datetime import datetime
import tensorboard
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Manager
from tensorflow.python.client import device_lib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Transformer(tf.keras.layers.Layer):
    def __init__(self):
        super(Transformer, self).__init__()

        self.attention1 = tf.keras.layers.MultiHeadAttention(num_heads=10, key_dim=10)
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.attention2 = tf.keras.layers.MultiHeadAttention(num_heads=10, key_dim=10)
        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.drop3 = tf.keras.layers.Dropout(0.1)
        self.drop4 = tf.keras.layers.Dropout(0.1)
        self.batch3 = tf.keras.layers.BatchNormalization()

    def call(self, src):
        data = src + 0

        att = self.attention1(data, data)
        data = data + self.drop1(att)
        data = self.batch1(data)
        att = self.attention2(data, data)
        data = data + self.drop2(att)
        data = self.batch2(data)
        att = self.dense2(self.drop3(self.dense1(data)))
        data = data + self.drop4(att)
        data = self.batch3(data)

        return data


class KINT(tf.keras.Model):
    def __init__(self):
        super(KINT, self).__init__()

        self.conv = tf.keras.layers.Conv1D(64, 9, padding='same', activation="relu")
        self.batch = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=3)

        self.transformer1 = Transformer()
        self.transformer2 = Transformer()
        self.transformer3 = Transformer()

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation=None)  # No activation on final dense layer
        self.l2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings

    def call(self, src):

        output = self.conv(src)
        output = self.batch(output)
        output = self.pool(output)

        output = self.transformer1(output)
        output = self.transformer2(output)
        output = self.transformer3(output)

        output = self.flat(output)

        output = self.dense1(output)
        output = self.dense2(output)
        output = self.l2(output)

        return output

    def build_graph(self):
        self.build((32, 501, 9))
        inputs = tf.keras.Input(shape=[501, 9])
        self.call(inputs)


def scheduler(epoch, lr):
    print('LEARNING RATE:', round(lr, 8))
    if epoch == 0:
        return lr
    else:
        return 0.0001


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, path, type, batch_size):

        self.batch_size = batch_size

        self.all_inputs = np.load(path + 'all_inputs_{}s.npy'.format(type))
        self.all_targets = np.load(path + 'all_targets_{}s.npy'.format(type))
        print(self.all_inputs.shape, self.all_targets.shape)
        self.on_epoch_end()

    def __getitem__(self, index):
        input = self.all_inputs[index * self.batch_size:(index + 1) * self.batch_size]
        target = self.all_targets[index * self.batch_size:(index + 1) * self.batch_size]
        target = np.argmax(target, axis=-1).squeeze()

        return input, target

    def on_epoch_end(self):
        randomize = np.arange(len(self.all_inputs))
        np.random.shuffle(randomize)
        self.all_inputs = self.all_inputs[randomize]
        self.all_targets = self.all_targets[randomize]

    def __len__(self):
        return math.floor(len(self.all_inputs) / self.batch_size)


def train():

    model = KINT()
    model.build_graph()
    # model.build((32, 500, 5))
    model(tf.keras.Input(shape=[501, 9]))
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[TruePositivesM()])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss(margin=0.2))
    print("First learning rate: ", round(model.optimizer.lr.numpy(), 5))
    model.summary()

    print("Making training dataset...", flush=True)
    train_d = DataGen(r'C:\HSE\EPISTASIS\nn\new_data_selected\\', 'train', 32)
    print("Making testing dataset...", flush=True)
    valid_d = DataGen(r'C:\HSE\EPISTASIS\nn\new_data_selected\\', 'test', 32)

    print("Training...", flush=True)

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = f"/home/avshmelev/bash_scripts/rnn/weights_frc_less_0.006.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 1 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,)
        # save_best_only=True,
        # monitor='val_true_positives_m',
        # mode='max')
        # save_freq=train_d.length*10)

    # Save the weights using the `checkpoint_path` format
    # model.save_weights(checkpoint_path.format(epoch=0))
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    # model.fit(train_d, validation_data=valid_d, epochs=100, batch_size=1, callbacks=[cp_callback, tensorboard_callback])
    the_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(train_d, validation_data=valid_d, epochs=60, batch_size=32, callbacks=[cp_callback, the_scheduler, tensorboard_callback], verbose=1)
    # model.save('/home/avshmelev/bash_scripts/rnn')
    print("Learning finished")


def test():

    model = KINT()
    model.build_graph()

    axis1 = 1
    axis2 = 120

    model.summary()
    model.load_weights(r"C:\home\avshmelev\bash_scripts\rnn\weights_frc_less_0.006.hdf5")

    input_test = np.load(r'C:\HSE\EPISTASIS\nn\new_data_selected\all_inputs_tests.npy')
    target_test = np.load(r'C:\HSE\EPISTASIS\nn\new_data_selected\all_targets_tests.npy')
    results_test = model.predict(input_test)

    input_train = np.load(r'C:\HSE\EPISTASIS\nn\new_data_selected\all_inputs_trains.npy')
    target_train = np.load(r'C:\HSE\EPISTASIS\nn\new_data_selected\all_targets_trains.npy')
    results_train = model.predict(input_train)

    px = 1 / plt.rcParams['figure.dpi']
    plt.clf()
    fig, ax = plt.subplots(5, 2, figsize=(6400 * px, 6400 * px))

    tmp_train = []
    tmp_test = []
    for i in range(10):
        for j, vec in enumerate(results_train):
            if np.argmax(target_train[j].squeeze()) == i:
                tmp_train.append(vec)
        tmp_train = np.array(tmp_train)
        projection_train = np.stack((tmp_train[:, axis1], tmp_train[:, axis2]), axis=-1)
        dots1 = ax[i % 5][i // 5].scatter(projection_train[:, 0], projection_train[:, 1], s=10, label='Train data')
        tmp_train = []

        for j, vec in enumerate(results_test):
            if np.argmax(target_test[j].squeeze()) == i:
                tmp_test.append(vec)
        tmp_test = np.array(tmp_test)
        projection_test = np.stack((tmp_test[:, axis1], tmp_test[:, axis2]), axis=-1)
        dots2 = ax[i % 5][i // 5].scatter(projection_test[:, 0], projection_test[:, 1], s=10, label='Test data')
        tmp_test = []

        ax[i % 5][i // 5].legend(handles=[dots1, dots2], fontsize=40)

    for i in range(10):
        ax[i % 5][i // 5].set_title(f"Class {i + 1}", fontsize=40)

    fig.suptitle(f"Projection for axes {axis1} and {axis2}", fontsize=80)
    fig.savefig("projection.png")

    # projection = np.stack((results_train[:, 128], results_train[:, 988]), axis=-1)
    # print(projection.shape)
    # plt.clf()
    # plt.scatter(projection[:, 0], projection[:, 1], s=0.1)
    # plt.show()

    clusters = np.zeros((10, 128))
    tmp = []
    for i in range(10):
        for j, vec in enumerate(results_train):
            if np.argmax(target_train[j].squeeze()) == i:
                tmp.append(vec)
        tmp = np.array(tmp)
        clusters[i, :] = tmp.mean(axis=0)
        tmp = []

    print(clusters)

    dist = np.zeros((10, 10))
    for i in range(10):
        for j in range(i + 1, 10):
            print(f"Distance between clusters {i} and {j} = {np.linalg.norm(clusters[i] - clusters[j])}")
            dist[i, j] = np.linalg.norm(clusters[i] - clusters[j])
            dist[j, i] = np.linalg.norm(clusters[i] - clusters[j])

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(dist)
    for (j, i), label in np.ndenumerate(dist):
        ax.text(i, j, round(label, 2), ha='center', va='center')
    fig.colorbar(img)
    fig.suptitle("Distances between clusters")
    plt.savefig("cluster_distances.png")

    # tsne = TSNE(learning_rate=10, perplexity=10)
    # embedded = tsne.fit_transform(results_train)
    # print('New Shape of X: ', embedded.shape)
    # print('Kullback-Leibler divergence after optimization: ', tsne.kl_divergence_)
    # print('No. of iterations: ', tsne.n_iter_)
    #
    # plt.clf()
    # fig, ax = plt.subplots(5, 2, figsize=(6400 * px, 6400 * px))
    # for i in range(10):
    #     for j, vec in enumerate(embedded):
    #         if np.argmax(target_train[j].squeeze()) == i:
    #             tmp_train.append(vec)
    #     tmp_train = np.array(tmp_train)
    #     dots = ax[i % 5][i // 5].scatter(tmp_train[:, 0], tmp_train[:, 1], s=10, label=f'Class {i}')
    #     tmp_train = []
    #     ax[i % 5][i // 5].legend(handles=[dots], fontsize=40)
    #
    # plt.savefig("New_clusters.png")

    # print(target_test[0])
    counter = 0
    for i in range(len(results_test)):
        # for i, vec in enumerate(results_test):
        # if i % 100 == 0:
        print(i)
        vec = results_test[i]
        real = target_test[i]
        norm_arg = np.argmin(np.linalg.norm(vec - results_train, axis=1))
        # print(np.min(np.linalg.norm(vec - results_train, axis=1)))
        # norm_arg = np.argmin(np.linalg.norm(vec - clusters, axis=1))
        answer = target_train[norm_arg]
        print(np.argmax(real.squeeze()), np.argmax(answer.squeeze()))
        class_list = np.argmax(target_train[np.argsort(np.linalg.norm(vec - results_train, axis=1))[:30]],
                               axis=-1).squeeze()
        print(class_list, "Mean class: ", np.mean(class_list))
        print(np.round(np.mean(class_list)))
        # if np.abs(np.argmax(real.squeeze()) - np.argmax(answer.squeeze())) <= 1:
        if np.abs(np.argmax(real.squeeze()) - np.round(np.mean(class_list))) <= 1:
            counter += 1

    print(f"Accuracy: {counter}/{len(results_test)}, exactly: {round(counter / len(results_test), 4)}")
    # norm_arg = np.argmin(np.linalg.norm(results_test[0] - results_train, axis=1))
    # print(target_train[norm_arg])


test()