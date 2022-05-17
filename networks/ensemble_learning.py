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


class KINT(tf.keras.Model):
    def __init__(self):
        super(KINT, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv5 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv6 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv7 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")
        self.conv8 = tf.keras.layers.Conv1D(16, 9, padding='same', activation="relu")

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
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, src):
        output = self.conv1(src)
        output = self.batch1(output)

        output = self.conv2(output)
        output = self.batch2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        output = self.batch3(output)

        output = self.conv4(output)
        output = self.batch4(output)

        output = self.pool2(output)

        output = self.conv5(output)
        output = self.batch5(output)

        output = self.conv6(output)
        output = self.batch6(output)

        output = self.pool3(output)

        output = self.conv7(output)
        output = self.batch7(output)

        output = self.conv8(output)
        output = self.batch8(output)

        output = self.flat(output)
        output = self.dense1(output)
        output = self.dense2(output)
        output = self.dense3(output)

        return output

    def build_graph(self):
        self.build((32, 1001, 7))
        inputs = tf.keras.Input(shape=[1001, 7])
        self.call(inputs)

def FormTarget(gen, frc):

    ########################################################### common
    gen_int = [(i + 1) * 100 for i in range(10)]

    frc_start = 0.001
    frc_end = 0.1
    l = np.log(frc_end) - np.log(frc_start)
    frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]
    print("FRC_CLASSES", np.trunc(np.array(frc_int) * 1000) / 1000)
    ###########################################################

    # adm_start = 0.0002
    # adm_end = 0.01
    # l = np.log(adm_end) - np.log(adm_start)
    # adm_int = [np.exp(l*(i+1)/10 + np.log(adm_start)) for i in range(10)]

    gen_array = np.full(10, 0)
    # adm_array = np.full(10, 0)
    # frc_array = np.full(10, 0)

    # adm_num = 0
    # frc_num = 0

    for k, elem in enumerate(gen_int):
        if gen == elem:
            gen_num = k

    # for k, elem in enumerate(adm_int):
    #   if adm > elem:
    #     adm_num = k + 1

    # for k, elem in enumerate(frc_int):
    #     if frc > elem:
    #         frc_num = k + 1

    gen_array[gen_num] = 1
    array = gen_array
    # if gen == 100:
    #     array = np.array([1, 0])
    #     sample_weights = np.array([10, 0])
    # else:
    #     array = np.array([0, 1])
    #     sample_weights = np.array([0, 1])
    # adm_array[adm_num] = 1
    # frc_array[frc_num] = 1
    # array = np.concatenate([gen_array, frc_array]).tolist()
    # array = torch.argmax(array).view(1).type(dtype=torch.float).cuda(0)

    return array.reshape((1, 10))

def scheduler(epoch, lr):
    print('LEARNING RATE:', round(lr, 8))
    if epoch == 0:
        return lr
    else:
        return 0.0001

def FormData(train_number, path):

    temp_data = []
    inputs = []
    targets = []
    sample_weights = []

    file_list = sorted(os.listdir(path))

    for i in range(train_number):
        if (i % 100) == 0:
            print("Train file ", i, flush=True)
        exact_file = path + '/' + file_list[i]
        file = open(exact_file, 'r')
        for line_num, line in enumerate(file):
            array = line.split('\t')
            temp = [array[2], array[3], array[4], array[5], array[6], array[7], array[8]]
            temp = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp]
            if temp[0] != 0:
                temp[0] = temp[0] * 10
            if temp[1] != 0:
                temp[1] = temp[1] * 1000
            if temp[2] != 0:
                temp[2] = temp[2] * 1000
            if temp[3] != 0:
                temp[3] = temp[3] * 1000000
            if temp[4] != 0:
                temp[4] = temp[4] * 1000000
            if temp[5] != 0:
                temp[5] = temp[5] * 1000
            if temp[6] != 0:
                temp[6] = temp[6] * 1000
            temp_data += [temp]
            if line_num == 1000:
                break
        # temp_data.append(float(exact_file.split('_')[4]))
        temp = np.array(temp_data)
        # start = temp[:500, :]
        # end = temp[:499:-1, :]
        # temp_data = np.add(start, end)
        # inputs += [temp_data]
        inputs += [temp]
        temp_data = []
        file.close()

        # one_target, one_sample_weight = FormTarget(float(exact_file.split('_')[3].split('/')[1]), float(exact_file.split('_')[5]))
        one_target = FormTarget(float(exact_file.split('_')[3].split('/')[1]), float(exact_file.split('_')[5]))
        targets += [one_target]
        # sample_weights.append(one_sample_weight)

    return np.array(inputs), np.array(targets) #, np.array(sample_weights).reshape((train_number, 1, 2))


class TruePositivesM(tf.keras.metrics.Accuracy):
    def __init__(self,
                 thresholds=None,
                 name=None,
                 dtype=None):
        super(TruePositivesM, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print(y_pred.shape)

        for i in range(32): ################################################################################################### change as batch size #############################
            # tf.print(i)
            # tf.print(y_pred[i:i+1])
            # tf.print(y_true[i:i + 1])
            t = tf.argmax(y_true[i:i+1], 1)
            p = tf.argmax(y_pred[i:i+1], 1)
            if tf.math.abs(t - p) <= tf.constant(1, dtype=tf.int64):
                t = tf.constant(1, dtype=tf.int64)
                p = tf.constant(1, dtype=tf.int64)
            else:
                t = tf.constant(1, dtype=tf.int64)
                p = tf.constant(0, dtype=tf.int64)
            super().update_state(t, p, sample_weight)

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, num, path, type, batch_size, frc_class, from_file=False):

        ########################################################### common
        gen_int = [(i + 1) * 100 for i in range(10)]

        frc_start = 0.001
        frc_end = 0.1
        l = np.log(frc_end) - np.log(frc_start)
        frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]
        print("FRC_CLASSES", np.trunc(np.array(frc_int) * 1000) / 1000)
        ###########################################################

        # adm_start = 0.0002
        # adm_end = 0.01
        # l = np.log(adm_end) - np.log(adm_start)
        # adm_int = [np.exp(l*(i+1)/10 + np.log(adm_start)) for i in range(10)]

        self.batch_size = batch_size
        self.length = None
        if type == "train":
            fixed_path = r"C:\HSE\EPISTASIS\nn\next_gen_simulation_max"
        if type == "test":
            fixed_path = r"C:\HSE\EPISTASIS\nn\next_gen_simulation_maxtest"
        file_list = sorted(os.listdir(fixed_path))
        file_list = list(map(lambda file_name: float(file_name.split("_")[2]), file_list))
        if frc_class == 0:
            selection = file_list <= frc_int[frc_class]
        else:
            selection = np.array(list(map(lambda frc: frc_int[frc_class - 1] < frc <= frc_int[frc_class], file_list)))
        if from_file:
            with open(path + 'all_inputs_{}.npy'.format(type), 'rb') as f:
                self.all_inputs = np.load(f)
                self.all_inputs = self.all_inputs[selection, :, :]
                self.all_inputs = self.all_inputs[::2, :, :]
                self.length = math.floor(len(self.all_inputs) / self.batch_size)
                # print((self.all_inputs[:, :, 0].squeeze() == 0.))
                # self.copy = np.all(self.all_inputs[:, 500, 0].squeeze() != 0.)
                # self.all_inputs = self.all_inputs[self.copy, :, :]
            with open(path + 'all_targets_{}.npy'.format(type), 'rb') as f:
                self.all_targets = np.load(f)
                self.all_targets = self.all_targets[selection, :, :]
                # self.all_targets = self.all_targets[self.copy, :, :]
            # with open('all_sample_weights_{}.npy'.format(type), 'rb') as f:
            #     self.all_sample_weights = np.load(f)
        else:
            # self.all_inputs, self.all_targets, self.all_sample_weights = FormData(num, path)
            self.all_inputs, self.all_targets = FormData(num, path)
            with open('all_inputs_{}.npy'.format(type), 'wb') as f:
                np.save(f, self.all_inputs)
            with open('all_targets_{}.npy'.format(type), 'wb') as f:
                np.save(f, self.all_targets)
            # with open('all_sample_weights_{}.npy'.format(type), 'wb') as f:
            #     np.save(f, self.all_sample_weights)
        print(self.all_inputs.shape, self.all_targets.shape)
        self.on_epoch_end()

    def __getitem__(self, index):
        input = self.all_inputs[index * self.batch_size:(index + 1) * self.batch_size]
        target = self.all_targets[index * self.batch_size:(index + 1) * self.batch_size]
        input = input.reshape((len(input), 1001, 7))
        # input = input - np.concatenate((input, input[:, -1:].reshape((self.batch_size, 1, 5))), axis=1)[:, 1:, :] ######################## возможно, это стоит убрать для FaceNet
        target = target.reshape((len(target), 10))
        # target = np.argmax(target, axis=-1).squeeze()
        # target = target.astype('float64')
        # for s in target:
        #     if np.argmax(s) != 0:
        #         s[np.argmax(s)-1:np.argmax(s)+2:2] = 0.99
        #     else:
        #         s[np.argmax(s):np.argmax(s)+2] = 0.99
        #     s[np.argmax(s)] = 1
        # tf.print(target)
        # target = np.array([0 if i != 1+3*np.argmax(target) else 1 for i in range(30)]).reshape(1, 30)
        # sample_weights = self.all_sample_weights[index].reshape(2)
        # print(target)

        return input, target

    def on_epoch_end(self):
        randomize = np.arange(len(self.all_inputs))
        np.random.shuffle(randomize)
        self.all_inputs = self.all_inputs[randomize]
        self.all_targets = self.all_targets[randomize]
        # self.all_sample_weights = self.all_sample_weights[randomize]

    def __len__(self):
        return math.floor(len(self.all_inputs) / self.batch_size)


def train(frc_class, net_count, acc):

    model = KINT()
    model.build_graph()
    # model.build((32, 500, 5))
    model(tf.keras.Input(shape=[1001, 7]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[TruePositivesM()])
    # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tfa.losses.TripletSemiHardLoss(margin=0.2))
    print("First learning rate: ", round(model.optimizer.lr.numpy(), 5))
    model.summary()

    print("Making training dataset...", flush=True)
    train_d = DataGen(96926, r'C:\HSE\EPISTASIS\nn\\', 'train', 32, frc_class, True)
    print("Making testing dataset...", flush=True)
    valid_d = DataGen(9693, r'C:\HSE\EPISTASIS\nn\\', 'test', 32, frc_class, True)

    print("Training...", flush=True)

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = f"/home/avshmelev/bash_scripts/rnn/weights_net_num_{net_count+1}_frc_class_{frc_class+1}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 1 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_true_positives_m',
        mode='max')
        # save_freq=train_d.length*10)

    # Save the weights using the `checkpoint_path` format
    # model.save_weights(checkpoint_path.format(epoch=0))
    # logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # model.fit(train_d, validation_data=valid_d, epochs=100, batch_size=1, callbacks=[cp_callback, tensorboard_callback])
    the_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(train_d, validation_data=valid_d, epochs=10, batch_size=32, callbacks=[cp_callback, the_scheduler], verbose=1)
    # model.save('/home/avshmelev/bash_scripts/rnn')
    acc.value = max(history.history['val_true_positives_m'])
    print("Learning finished", flush=True)


def test(l, path, model_weights, all_inputs):
    length = len(all_inputs)
    model = KINT()
    model.build_graph()
    model.load_weights(path + model_weights)
    l[:] = model.predict(all_inputs).flatten().tolist()


if __name__ == "__main__":

    t = True

    # tensorflow stats
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print(device_lib.list_local_devices())  # list of DeviceAttributes
    print(tf.version.VERSION)

    if t:
        plt.clf()
        for frc_class in range(10):
            accuracies = []
            acc = Value('d', 0.0)
            for net_count in range(2):
                print(f" Class: {frc_class+1}, Net Number: {net_count+1} ".center(100, "~"))
                p = Process(target=train, args=(frc_class, net_count, acc))
                p.start()
                p.join()
                accuracies.append(acc.value)
            plt.scatter(list(range(1, 3)), accuracies, label=f"Force class {frc_class}", alpha=0.3, s=6, edgecolors='none')
        plt.legend()
        plt.grid(True)
        plt.savefig("ensemble_stats.png")
    else:
        path = r"C:\home\avshmelev\bash_scripts\rnn\\"
        file_list = sorted(os.listdir(path))
        predictions = []
        with open('all_inputs_test.npy', 'rb') as f:
            all_inputs = np.load(f)
        with open('all_targets_test.npy', 'rb') as f:
            all_targets = np.load(f)
        for iter, model_weights in enumerate(file_list):
            with Manager() as manager:
                l = manager.list(range(10*len(all_inputs)))
                p = Process(target=test, args=(l, path, model_weights, all_inputs))
                p.start()
                p.join()
                predictions.append(np.array(l).reshape(len(all_inputs), 10))
                print(np.array(predictions)[iter][0])

        with open('all_predictions.npy', 'wb') as f:
            np.save(f, np.array(predictions))





























def test_old():
    def DataGen():
        with open('all_inputs_test.npy', 'rb') as f:
            all_inputs = np.load(f)
        with open('all_targets_test.npy', 'rb') as f:
            all_targets = np.load(f)

        all_inputs = all_inputs - np.concatenate((all_inputs, all_inputs[:, -1:].reshape((10199, 1, 5))), axis=1)[:, 1:, :]

        print(all_inputs[0])
        print(all_targets[0])

        return all_inputs, all_targets

    class TruePositivesM(tf.keras.metrics.Accuracy):
        def __init__(self,
                     thresholds=None,
                     name=None,
                     dtype=None):
            super(TruePositivesM, self).__init__(name=name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.argmax(y_true, 1)
            y_pred = tf.argmax(y_pred, 1)
            if tf.math.abs(y_true - y_pred) <= tf.constant(1, dtype=tf.int64):
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(1, dtype=tf.int64)
            else:
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(0, dtype=tf.int64)
            super().update_state(y_true, y_pred, sample_weight)

    class KINT(tf.keras.Model):
        def __init__(self):
            super(KINT, self).__init__()
            self.activation = tf.keras.layers.Activation(tf.nn.relu)

            self.attention1 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=20)
            self.attention2 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=20)
            self.attention3 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=20)
            self.attention4 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=20)
            self.conv1 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv2 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv3 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv4 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv5 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv6 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv7 = tf.keras.layers.Conv1D(256, 9, padding='same')
            self.conv8 = tf.keras.layers.Conv1D(256, 9, padding='same')

            self.batch1 = tf.keras.layers.BatchNormalization()
            self.batch2 = tf.keras.layers.BatchNormalization()
            self.batch3 = tf.keras.layers.BatchNormalization()
            self.batch4 = tf.keras.layers.BatchNormalization()
            self.batch5 = tf.keras.layers.BatchNormalization()
            self.batch6 = tf.keras.layers.BatchNormalization()
            self.batch7 = tf.keras.layers.BatchNormalization()
            self.batch8 = tf.keras.layers.BatchNormalization()

            self.batch9 = tf.keras.layers.BatchNormalization()
            self.batch10 = tf.keras.layers.BatchNormalization()
            self.batch11 = tf.keras.layers.BatchNormalization()
            self.batch12 = tf.keras.layers.BatchNormalization()
            self.batch13 = tf.keras.layers.BatchNormalization()
            self.batch14 = tf.keras.layers.BatchNormalization()

            self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
            self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
            self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=5)

            self.dense1 = tf.keras.layers.Dense(2000)
            self.dense2 = tf.keras.layers.Dense(1000)
            self.dense3 = tf.keras.layers.Dense(500)
            self.dense4 = tf.keras.layers.Dense(500)
            self.dense5 = tf.keras.layers.Dense(500)
            self.dense6 = tf.keras.layers.Dense(250)
            self.dense7 = tf.keras.layers.Dense(10, activation="sigmoid")


        def call(self, src):
            # here will be outputs of the each layer
            output = self.conv1(src)  # (32, 500, 1024)
            output = self.batch1(output)
            output = self.activation(output)
            output = self.conv2(output)
            output = self.batch2(output)
            output = self.activation(output)
            output = self.pool1(output)  # (32, 250, 1024)
            output = self.attention1(output, output)

            output = self.conv3(output)  # (32, 250, 1024)
            output = self.batch3(output)
            output = self.activation(output)
            output = self.conv4(output)
            output = self.batch4(output)
            output = self.activation(output)
            output = self.pool2(output)  # (32, 125, 1024)
            output = self.attention2(output, output)

            output = self.conv5(output)  # (32, 125, 1024)
            output = self.batch5(output)
            output = self.activation(output)
            output = self.conv6(output)
            output = self.batch6(output)
            output = self.activation(output)
            output = self.pool3(output)  # (32, 25, 1024)
            output = self.attention3(output, output)

            output = self.conv7(output)  # (32, 25, 1024)
            output = self.batch7(output)
            output = self.activation(output)
            output = self.conv8(output)
            output = self.batch8(output)
            output = self.activation(output)
            output = self.attention4(output, output)

            output = tf.reshape(output, [1, 6400])

            output = self.dense1(output)
            output = self.batch9(output)
            output = self.activation(output)
            output = self.dense2(output)
            output = self.batch10(output)
            output = self.activation(output)
            output = self.dense3(output)
            output = self.batch11(output)
            output = self.activation(output)
            output = self.dense4(output)
            output = self.batch12(output)
            output = self.activation(output)
            output = self.dense5(output)
            output = self.batch13(output)
            output = self.activation(output)
            output = self.dense6(output)
            output = self.batch14(output)
            output = self.activation(output)
            output = self.dense7(output)

            # output = tf.reshape(output, [2, 10])
            # output = tf.nn.softmax(output)
            # output = tf.keras.activations.sigmoid(output)

            return output

        def build_graph(self):
            self.build((1, 500, 5))
            inputs = tf.keras.Input(shape=[500, 5])
            self.call(inputs)

    with tf.device('/cpu:0'):
        model = KINT()
        model.build_graph()
        # model.build((32, 500, 5))
        model(tf.keras.Input(shape=[500, 5]))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=[TruePositivesM()])
        model.summary()
        model.load_weights("weights.180.hdf5")
        print("Making testing dataset...", flush=True)
        input, target = DataGen()
        target = target.reshape((len(target), 10))
        counter = 0
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        print(model.layers[30].get_weights()[0].shape)
        # img = ax.imshow(model.layers[1].get_weights()[0][:, 1, :])
        # img = ax.imshow(model.layers[12].get_weights()[0][:, :, 200])
        img = ax.imshow(model.layers[30].get_weights()[0])
        fig.colorbar(img)
        plt.show()
        # print("MEMORY: {}".format(tf.config.experimental.get_memory_info('/cpu:0')))
        # for i in range(10199):
        #     # print("Answer: {}".format(i))
        #     prediction = model(input[i].reshape((1, 500, 5))).numpy() ############# в переменной prediction хранится массив вероятностей
        #     print(prediction)
        #     print(target[i])
        #     max_predicted = np.argmax(prediction.flatten())
        #     max_true = np.argmax(target[i].reshape((1, 10)))
        #     if abs(max_predicted - max_true) <= 2:
        #         counter += 1
        #         print("Correct")
        #     else:
        #         print(i)
        # print(counter)
    # loss, acc = model.evaluate(input, target, verbose=2, batch_size=1)
    # prediction = model.predict(input, verbose=1, batch_size=1)
    # print(prediction)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
# p1 = Process(target=INT)
# p1.start()
# # p2 = Process(target=GEN)
# # p2.start()
# p1.join()
# # p2.join()

# INT()
# test()

# input_shape = (1, 2, 3)
# x = tf.random.normal(input_shape)
# print(x)
# y = tf.keras.layers.Conv1D(2, 1, padding="same", kernel_initializer=tf.keras.initializers.Constant(1.))(x)
# # y = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
# print(y)

# x = np.arange(10).reshape(1, 5, 2)
# print(x)
#
# y = np.arange(10, 20).reshape(1, 2, 5)
# print(y)
#
# print(tf.keras.layers.Dot(axes=(1, 2))([y, x]))
