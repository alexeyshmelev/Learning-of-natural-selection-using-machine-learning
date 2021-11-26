import os
import sys
import time
import torch
import math
import settings  # our custom file
import numpy as np
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
from multiprocessing import Process
from torch.utils.data import TensorDataset, DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# на даный момент сеть не предназначена для работы с участками естественного отбора, идущими НЕ через 1 (т.е. допустим 0.00, 0.02, 0.04 ... - не подходит)

num_classes = settings.num_classes
input_size = settings.input_size
hidden_size = settings.hidden_size
batch_size = settings.batch_size
sequence_length = settings.sequence_length
num_layers = settings.num_layers
train_number = settings.train_number
dim = settings.dim
line_format = '%.' + str(dim) + 'f'

def INT():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())  # list of DeviceAttributes

    print(tf.version.VERSION)

    gen_int = [(i + 1) * 100 for i in range(10)]

    # adm_start = 0.0002
    # adm_end = 0.01
    # l = np.log(adm_end) - np.log(adm_start)
    # adm_int = [np.exp(l*(i+1)/10 + np.log(adm_start)) for i in range(10)]

    frc_start = 0.001
    frc_end = 0.1
    l = np.log(frc_end) - np.log(frc_start)
    frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]

    def FormTarget(gen, frc):

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
        # adm_array[adm_num] = 1
        # frc_array[frc_num] = 1
        # array = np.concatenate([gen_array, frc_array]).tolist()
        # array = torch.argmax(array).view(1).type(dtype=torch.float).cuda(0)

        return array

    def FormData(train_number, path):

        temp_data = []
        inputs = []
        targets = []

        for i in range(train_number):
            if (i % 100) == 0:
                print("Train file ", i, flush=True)
            file_list = sorted(os.listdir(path))
            exact_file = path + '/' + file_list[i]
            file = open(exact_file, 'r')
            for line in file:
                array = line.split('\t')
                temp = [array[2], array[3], array[4], array[5], array[6]]
                temp = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp]
                if temp[0] != 0:
                    temp[0] = np.log(temp[0] * 10) / np.log(1.6)
                if temp[1] != 0:
                    temp[1] = np.log(temp[1] * 100) / np.log(1.6)
                if temp[2] != 0:
                    temp[2] = np.log(temp[2] * 100) / np.log(1.6)
                if temp[3] != 0:
                    temp[3] = np.log(temp[3] * 1000000) / np.log(1.6)
                if temp[4] != 0:
                    temp[4] = np.log(temp[4] * 1000000) / np.log(1.6)
                temp_data.append(temp)
            # temp_data.append(float(exact_file.split('_')[4]))
            temp = np.array(temp_data).reshape(1000, 5)
            start = temp[:500, :]
            end = temp[:499:-1, :]
            temp_data = np.add(start, end).flatten().tolist()
            inputs.append(temp_data)
            temp_data = []
            file.close()

            targets.append(FormTarget(float(exact_file.split('_')[3].split('/')[1]), float(exact_file.split('_')[5])))

        return np.array(inputs).reshape((train_number, 500, 5)), np.array(targets).reshape((train_number, 1, 10))

    class DataGen(tf.keras.utils.Sequence):
        def __init__(self, num, path, type, from_file=False):
            if from_file:
                with open('all_inputs_{}.npy'.format(type), 'rb') as f:
                    self.all_inputs = np.load(f)
                with open('all_targets_{}.npy'.format(type), 'rb') as f:
                    self.all_targets = np.load(f)
            else:
                self.all_inputs, self.all_targets = FormData(num, path)
                with open('all_inputs_{}.npy'.format(type), 'wb') as f:
                    np.save(f, self.all_inputs)
                with open('all_targets_{}.npy'.format(type), 'wb') as f:
                    np.save(f, self.all_targets)
            print(self.all_inputs.shape, self.all_targets.shape)
            self.on_epoch_end()

        def __getitem__(self, index):
            input = self.all_inputs[index].reshape(1, 500, 5)
            target = self.all_targets[index].reshape(1, 10)
            input = (input, target.reshape((1, 10, 1)).astype(float))

            return input, target

        def on_epoch_end(self):
            randomize = np.arange(len(self.all_inputs))
            np.random.shuffle(randomize)
            self.all_inputs = self.all_inputs[randomize]
            self.all_targets = self.all_targets[randomize]

        def __len__(self):
            return len(self.all_inputs)

    class TruePositivesM(tf.keras.metrics.Accuracy):
        def __init__(self,
                     thresholds=None,
                     name=None,
                     dtype=None):
            super(TruePositivesM, self).__init__(name=name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.argmax(y_true, 1)
            y_pred = tf.argmax(y_pred, 1)
            super().update_state(y_true, y_pred, sample_weight)

    class KINT(tf.keras.Model):
        def __init__(self):
            super(KINT, self).__init__()

            self.attention1 = tf.keras.layers.MultiHeadAttention(num_heads=20, key_dim=9)

            # self.norm1_1 = tf.keras.layers.BatchNormalization()

            # self.conv1_1 = tf.keras.layers.Conv2D(64, (5, 9), padding='same', activation="relu")

            # self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))

            # self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150))

            self.dense = tf.keras.layers.Dense(10, activation="softmax")

        def call(self, src):
            # here will be outputs of the each layer
            # output = tf.expand_dims(src, axis=3)  # (1, 500, 5, 1)
            output = self.attention1(src[0], src[1])

            output = tf.reshape(output, [1, 2500])
            output = self.dense(output)

            # output = tf.reshape(output, [2, 10])
            # output = tf.nn.softmax(output)
            # output = tf.keras.activations.sigmoid(output)

            return output

    if settings.boot_from_file:
        pass

    else:
        model = KINT()
        model((tf.keras.Input(shape=[500, 5]), tf.keras.Input(shape=[10, 1])))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[TruePositivesM()])
        model.summary()

        print("Making training dataset...", flush=True)
        # train_d = DataGen(11128, 'next_gen_simulation_final')
        train_d = DataGen(111993, 'next_gen_simulation_usa', 'train', True)
        print("Making testing dataset...", flush=True)
        valid_d = DataGen(199, 'next_gen_simulation_usatest', 'test', True)

        print("Training...", flush=True)

        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = "weights.{epoch:02d}.hdf5"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        batch_size = 111993

        # Create a callback that saves the model's weights every 1 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=1 * batch_size)

        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path.format(epoch=0))

        model.fit(train_d, validation_data=valid_d, epochs=10, callbacks=[cp_callback])
        # model.save('/home/avshmelev/bash_scripts/rnn')
        print("Learning finished", flush=True)





























def test():
    def DataGen():
        with open('all_inputs_test.npy', 'rb') as f:
            all_inputs = np.load(f)
        with open('all_targets_test.npy', 'rb') as f:
            all_targets = np.load(f)

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
            super().update_state(y_true, y_pred, sample_weight)

    class KINT(tf.keras.Model):
        def __init__(self):
            super(KINT, self).__init__()
            self.attention1 = tf.keras.layers.Attention()
            self.attention2 = tf.keras.layers.Attention()
            self.attention3 = tf.keras.layers.Attention()
            self.conv1 = tf.keras.layers.Conv1D(64, 20, padding='same', activation="relu")
            self.conv2 = tf.keras.layers.Conv1D(64, 20, padding='same')
            self.conv3 = tf.keras.layers.Conv1D(64, 20, padding='same', activation="relu")
            self.conv4 = tf.keras.layers.Conv1D(64, 20, padding='same')
            self.conv5 = tf.keras.layers.Conv1D(64, 20, padding='same', activation="relu")
            self.conv6 = tf.keras.layers.Conv1D(64, 20, padding='same')
            self.conv7 = tf.keras.layers.Conv1D(64, 20, padding='same', activation="relu")
            self.conv8 = tf.keras.layers.Conv1D(64, 20, padding='same')
            self.conv9 = tf.keras.layers.Conv1D(64, 20, padding='same', activation="relu")
            self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
            self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
            self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

            self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

        def call(self, src):
            output = self.attention1([src, src])
            output = self.conv1(output)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.rnn1(output)

            output = self.attention2([output, output])
            output = self.conv4(output)
            output = self.conv5(output)
            output = self.conv6(output)
            output = self.rnn2(output)

            output = self.attention3([output, output])
            output = self.conv7(output)
            output = self.conv8(output)
            output = self.conv9(output)
            output = self.rnn3(output)

            output = self.dense1(output)

            # output = tf.reshape(output, [2, 10])
            # output = tf.nn.softmax(output)
            # output = tf.reshape(output, [1, 10])
            # output = tf.keras.activations.sigmoid(output)

            return output

    model = KINT()
    model.build((None, 1000, 5))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[TruePositivesM()])
    model.summary()
    model.load_weights("/home/avshmelev/bash_scripts/rnn/weights.05.hdf5")
    print("Making testing dataset...", flush=True)
    input, target = DataGen()
    counter = 0
    for i in range(199):
        # print("Answer: {}".format(i))
        prediction = model(input[i].reshape((1, 1000, 5)))
        max_predicted = np.argmax(prediction.numpy().flatten())
        max_true = np.argmax(target[i])
        if max_predicted == max_true:
            counter += 1
    print(counter)
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

INT()
# test()