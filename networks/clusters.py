import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numba import prange, njit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'


class KINT(tf.keras.Model):
    def __init__(self):
        super(KINT, self).__init__()
        # self.activation = tf.keras.layers.Activation(tf.nn.relu)

        # self.attention1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=20)
        # self.attention2 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=20)
        # self.attention3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=20)

        self.conv1 = tf.keras.layers.Conv1D(128, 20, padding='same', activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(128, 20, padding='same', activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(128, 20, padding='same', activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(128, 20, padding='same', activation="relu")
        self.conv5 = tf.keras.layers.Conv1D(256, 15, padding='same', activation="relu")
        self.conv6 = tf.keras.layers.Conv1D(256, 15, padding='same', activation="relu")
        self.conv7 = tf.keras.layers.Conv1D(256, 15, padding='same', activation="relu")
        self.conv8 = tf.keras.layers.Conv1D(256, 15, padding='same', activation="relu")
        self.conv9 = tf.keras.layers.Conv1D(512, 10, padding='same', activation="relu")
        self.conv10 = tf.keras.layers.Conv1D(512, 10, padding='same', activation="relu")
        self.conv11 = tf.keras.layers.Conv1D(512, 10, padding='same', activation="relu")
        self.conv12 = tf.keras.layers.Conv1D(512, 10, padding='same', activation="relu")
        self.conv13 = tf.keras.layers.Conv1D(1024, 5, padding='same', activation="relu")
        self.conv14 = tf.keras.layers.Conv1D(1024, 5, padding='same', activation="relu")
        self.conv15 = tf.keras.layers.Conv1D(1024, 5, padding='same', activation="relu")
        self.conv16 = tf.keras.layers.Conv1D(1024, 5, padding='same', activation="relu")

        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.batch4 = tf.keras.layers.BatchNormalization()
        self.batch5 = tf.keras.layers.BatchNormalization()
        self.batch6 = tf.keras.layers.BatchNormalization()
        self.batch7 = tf.keras.layers.BatchNormalization()
        self.batch8 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=5)

        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
        # self.dense3 = tf.keras.layers.Dense(10, activation="softmax")
        self.dense3 = tf.keras.layers.Dense(128, activation=None)  # No activation on final dense layer
        self.l2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings

    def call(self, src):
        # here will be outputs of the each layer
        output = self.conv1(src)  # (32, 500, 1024)
        output = self.batch1(output)

        output = self.conv2(output)
        output = self.batch2(output)

        # output = self.attention1(output, output)
        output = self.pool1(output)  # (32, 250, 1024)

        output = self.conv3(output)  # (32, 250, 1024)
        output = self.batch3(output)

        output = self.conv4(output)
        output = self.batch4(output)

        # output = self.attention2(output, output)
        output = self.pool2(output)  # (32, 125, 1024)

        output = self.conv5(output)  # (32, 125, 1024)
        output = self.batch5(output)

        output = self.conv6(output)
        output = self.batch6(output)

        # output = self.attention3(output, output)
        output = self.pool3(output)  # (32, 25, 1024)

        output = self.conv7(output)  # (32, 25, 1024)
        output = self.batch7(output)

        output = self.conv8(output)
        output = self.batch8(output)

        # output = tf.reshape(output, [32, 1600])

        output = self.flat(output)
        output = self.dense1(output)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.l2(output)

        # output = tf.reshape(output, [2, 10])
        # output = tf.nn.softmax(output)
        # output = tf.keras.activations.softmax(output)

        return output

    def build_graph(self):
        self.build((32, 500, 5))
        inputs = tf.keras.Input(shape=[500, 5])
        self.call(inputs)


def DataGen(test=True):
    if test:
        with open(r'C:\HSE\EPISTASIS\nn\new_data\all_inputs_test.npy', 'rb') as f:
            all_inputs = np.load(f)
            copy = np.all(all_inputs[:, :, 0].squeeze() != 0, axis=1)
            all_inputs = all_inputs[copy, :, :]
        with open(r'C:\HSE\EPISTASIS\nn\new_data\all_targets_test.npy', 'rb') as f:
            all_targets = np.load(f)
            all_targets = all_targets[copy, :, :]

        # all_inputs = all_inputs - np.concatenate((all_inputs, all_inputs[:, -1:].reshape((10199, 1, 5))), axis=1)[:, 1:, :]
    else:
        with open(r'C:\HSE\EPISTASIS\nn\old_data\all_inputs_train.npy', 'rb') as f:
            all_inputs = np.load(f)
        with open(r'C:\HSE\EPISTASIS\nn\old_data\all_targets_train.npy', 'rb') as f:
            all_targets = np.load(f)

        # all_inputs = all_inputs - np.concatenate((all_inputs, all_inputs[:, -1:].reshape((111993, 1, 5))), axis=1)[:, 1:, :]

    print(all_inputs[0])
    print(all_targets[0])

    return all_inputs, all_targets


if __name__ == '__main__':

    file_list = sorted(os.listdir(r"C:\HSE\EPISTASIS\nn\next_gen_simulation_usatest_big"))

    model = KINT()
    model.build_graph()
    # model.build((32, 500, 5))

    model.summary()
    model.load_weights("weights.76.hdf5")

    axis1 = 1
    axis2 = 120

    input_test, target_test = DataGen()
    results_test = model.predict(input_test)

    input_train, target_train = DataGen(test=False)
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
        ax[i % 5][i // 5].set_title(f"Class {i+1}", fontsize=40)

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
        for j in range(i+1, 10):
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
    for i in prange(len(results_test)):
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
        class_list = np.argmax(target_train[np.argsort(np.linalg.norm(vec - results_train, axis=1))[:30]], axis=-1).squeeze()
        print(class_list, file_list[i], "Mean class: ", np.mean(class_list))
        print(np.round(np.mean(class_list)))
        # if np.abs(np.argmax(real.squeeze()) - np.argmax(answer.squeeze())) <= 1:
        if np.abs(np.argmax(real.squeeze()) - np.round(np.mean(class_list))) <= 1:
            counter += 1

    print(f"Accuracy: {counter}/{len(results_test)}, exactly: {round(counter/len(results_test), 4)}")
    # norm_arg = np.argmin(np.linalg.norm(results_test[0] - results_train, axis=1))
    # print(target_train[norm_arg])