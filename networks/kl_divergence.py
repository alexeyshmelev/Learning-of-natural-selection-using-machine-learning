import os
import sys
import time
import torch
import math
from datetime import datetime
import tensorboard
import settings  # our custom file
import numpy as np
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
from multiprocessing import Process
from torch.utils.data import TensorDataset, DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
            for line in file:
                array = line.split('\t')
                temp = [array[2], array[3], array[4], array[5], array[6]]
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
                temp_data += [temp]
            # temp_data.append(float(exact_file.split('_')[4]))
            temp = np.array(temp_data)
            start = temp[:500, :]
            end = temp[:499:-1, :]
            temp_data = np.add(start, end)
            inputs += [temp_data]
            temp_data = []
            file.close()

            # one_target, one_sample_weight = FormTarget(float(exact_file.split('_')[3].split('/')[1]), float(exact_file.split('_')[5]))
            one_target = FormTarget(float(exact_file.split('_')[3].split('/')[1]), float(exact_file.split('_')[5]))
            targets += [one_target]
            # sample_weights.append(one_sample_weight)

        return np.array(inputs), np.array(targets) #, np.array(sample_weights).reshape((train_number, 1, 2))

    class DataGen(tf.keras.utils.Sequence):
        def __init__(self, num, path, type, from_file=False):
            if from_file:
                with open('all_inputs_{}.npy'.format(type), 'rb') as f:
                    self.all_inputs = np.load(f)
                with open('all_targets_{}.npy'.format(type), 'rb') as f:
                    self.all_targets = np.load(f)
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
            input = self.all_inputs[index].reshape(1, 500, 5)
            target = self.all_targets[index].reshape(1, 10)
            # target = np.array([np.argmax(target)]).reshape((1, 1))
            # target = np.array([0 if i != 1+3*np.argmax(target) else 1 for i in range(30)]).reshape(1, 30)
            # sample_weights = self.all_sample_weights[index].reshape(2)

            return input, target

        def on_epoch_end(self):
            randomize = np.arange(len(self.all_inputs))
            np.random.shuffle(randomize)
            self.all_inputs = self.all_inputs[randomize]
            self.all_targets = self.all_targets[randomize]
            # self.all_sample_weights = self.all_sample_weights[randomize]

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
            # y_true = tf.cast(tf.reduce_max(y_true, 1), dtype=tf.int64)
            # y_pred = tf.cast(tf.round(tf.reduce_max(y_pred, 1)), dtype=tf.int64)
            if tf.math.abs(y_true - y_pred) <= tf.constant(1, dtype=tf.int64):
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(1, dtype=tf.int64)
            else:
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(0, dtype=tf.int64)
            super().update_state(y_true, y_pred, sample_weight)

    class WAY(tf.keras.layers.Layer):
        def __init__(self):
            super(WAY, self).__init__()

            self.initial1 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial2 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial_pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial3 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial4 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial_pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial5 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial6 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial_pool3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.initial7 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial8 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial_pool4 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.reshape = tf.keras.layers.Reshape((1280,))

            self.dense = tf.keras.layers.Dense(1)

        def call(self, src):
            # here will be outputs of the each layer
            # input: (1, 500, 5)
            output = self.initial1(src)  # output: (1, 500, 32)
            output = self.initial2(output)  # output: (1, 500, 32)
            output = self.initial_pool1(output)  # output: (1, 250, 32)

            output = self.initial3(output)  # output: (1, 250, 64)
            output = self.initial4(output)  # output: (1, 250, 64)
            output = self.initial_pool2(output)  # output: (1, 125, 64)

            output = self.initial5(output)  # output: (1, 125, 128)
            output = self.initial6(output)  # output: (1, 125, 128)
            output = self.initial_pool3(output)  # output: (1, 25, 128)

            output = self.initial7(output)  # output: (1, 25, 256)
            output = self.initial8(output)  # output: (1, 25, 256)
            output = self.initial_pool4(output)  # output: (1, 5, 256)

            output = self.reshape(output)

            output = self.dense(output)

            return output

    class CNN_Attention(tf.keras.layers.Layer):
        def __init__(self):
            super(CNN_Attention, self).__init__()

            self.dot = tf.keras.layers.Dot(axes=(1, 2))
            self.activation = tf.keras.layers.Activation('softmax')

        def call(self, src):

            output = self.dot([src, tf.transpose(src, perm=[0, 2, 1])])
            output = self.activation(output)
            output = self.dot([tf.transpose(src, perm=[0, 2, 1]), output]) + src

            return output

    class KINT(tf.keras.Model):
        def __init__(self):
            super(KINT, self).__init__()

            self.initial1 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial2 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial_pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial3 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial4 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial_pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial5 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial6 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial_pool3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.initial7 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial8 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial_pool4 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.reshape = tf.keras.layers.Reshape((160,))

            self.drop1 = tf.keras.layers.Dropout(0.)
            self.drop2 = tf.keras.layers.Dropout(0.)
            self.drop3 = tf.keras.layers.Dropout(0.)

            self.dense1 = tf.keras.layers.Dense(1000, activation="relu")
            self.dense2 = tf.keras.layers.Dense(500, activation="relu")
            self.dense3 = tf.keras.layers.Dense(250, activation="relu")
            self.dense4 = tf.keras.layers.Dense(125, activation="relu")
            self.dense5 = tf.keras.layers.Dense(75, activation="relu")
            self.dense6 = tf.keras.layers.Dense(10, activation="softmax")

        def call(self, src):
            # here will be outputs of the each layer
            # input: (1, 500, 5)
            output = self.initial1(src)  # output: (1, 500, 32)
            output = self.initial2(output)  # output: (1, 500, 32)
            output = self.initial_pool1(output)  # output: (1, 250, 32)

            output = self.initial3(output)  # output: (1, 250, 64)
            output = self.initial4(output)  # output: (1, 250, 64)
            output = self.initial_pool2(output)  # output: (1, 125, 64)

            output = self.initial5(output)  # output: (1, 125, 128)
            output = self.initial6(output)  # output: (1, 125, 128)
            output = self.initial_pool3(output)  # output: (1, 25, 128)

            output = self.initial7(output)  # output: (1, 25, 256)
            output = self.initial8(output)  # output: (1, 25, 256)
            output = self.initial_pool4(output)  # output: (1, 5, 256)

            output = self.reshape(output)

            output = self.drop1(self.dense1(output))
            output = self.drop2(self.dense2(output))
            output = self.drop3(self.dense3(output))
            output = self.dense4(output)
            output = self.dense5(output)
            output = self.dense6(output)

            return output

    if settings.boot_from_file:
        pass

    else:
        model = KINT()
        # model.build((None, 500, 5))
        model(tf.keras.Input(shape=[500, 5]))
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tf.keras.losses.KLDivergence(), metrics=[TruePositivesM()])
        print(round(model.optimizer.lr.numpy(), 5))
        model.summary()

        print("Making testing dataset...", flush=True)
        valid_d = DataGen(199, 'next_gen_simulation_usatest', 'test', True)
        print("Making training dataset...", flush=True)
        # train_d = DataGen(11128, 'next_gen_simulation_final')
        train_d = DataGen(111993, 'next_gen_simulation_usa', 'train', True)

        print("Training...", flush=True)

        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = "/home/avshmelev/bash_scripts/rnn/weights.{epoch:02d}.hdf5"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every 1 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=111993)

        # Save the weights using the `checkpoint_path` format
        # model.save_weights(checkpoint_path.format(epoch=0))
        # logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        # model.fit(train_d, validation_data=valid_d, epochs=100, batch_size=1, callbacks=[cp_callback, tensorboard_callback])
        the_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        model.fit(train_d, validation_data=valid_d, epochs=100, batch_size=1, callbacks=[cp_callback, the_scheduler])
        print(round(model.optimizer.lr.numpy(), 5))
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
            if tf.math.abs(y_true - y_pred) <= tf.constant(1, dtype=tf.int64):
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(1, dtype=tf.int64)
            else:
                y_true = tf.constant(1, dtype=tf.int64)
                y_pred = tf.constant(0, dtype=tf.int64)
            super().update_state(y_true, y_pred, sample_weight)

    class WAY(tf.keras.layers.Layer):
        def __init__(self):
            super(WAY, self).__init__()

            self.initial1 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial2 = tf.keras.layers.Conv1D(32, 2, padding='same', activation="relu")
            self.initial_pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial3 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial4 = tf.keras.layers.Conv1D(64, 2, padding='same', activation="relu")
            self.initial_pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

            self.initial5 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial6 = tf.keras.layers.Conv1D(128, 2, padding='same', activation="relu")
            self.initial_pool3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.initial7 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial8 = tf.keras.layers.Conv1D(256, 2, padding='same', activation="relu")
            self.initial_pool4 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)

            self.reshape = tf.keras.layers.Reshape((1280,))

            self.dense = tf.keras.layers.Dense(1)

        def call(self, src):
            # here will be outputs of the each layer
            # input: (1, 500, 5)
            output = self.initial1(src)  # output: (1, 500, 32)
            output = self.initial2(output)  # output: (1, 500, 32)
            output = self.initial_pool1(output)  # output: (1, 250, 32)

            output = self.initial3(output)  # output: (1, 250, 64)
            output = self.initial4(output)  # output: (1, 250, 64)
            output = self.initial_pool2(output)  # output: (1, 125, 64)

            output = self.initial5(output)  # output: (1, 125, 128)
            output = self.initial6(output)  # output: (1, 125, 128)
            output = self.initial_pool3(output)  # output: (1, 25, 128)

            output = self.initial7(output)  # output: (1, 25, 256)
            output = self.initial8(output)  # output: (1, 25, 256)
            output = self.initial_pool4(output)  # output: (1, 5, 256)

            output = self.reshape(output)

            output = self.dense(output)

            return output

    class KINT(tf.keras.Model):
        def __init__(self):
            super(KINT, self).__init__()

            self.way1 = WAY()
            self.way2 = WAY()
            self.way3 = WAY()
            self.way4 = WAY()
            self.way5 = WAY()
            self.way6 = WAY()
            self.way7 = WAY()
            self.way8 = WAY()
            self.way9 = WAY()
            self.way10 = WAY()

            self.concat = tf.keras.layers.Concatenate()

            self.dense1 = tf.keras.layers.Dense(1000, activation="relu")
            self.dense2 = tf.keras.layers.Dense(500, activation="relu")
            self.dense3 = tf.keras.layers.Dense(250, activation="relu")
            self.dense4 = tf.keras.layers.Dense(150, activation="relu")
            self.dense5 = tf.keras.layers.Dense(75, activation="relu")
            self.dense6 = tf.keras.layers.Dense(10, activation="softmax")


        def call(self, src):

            output1 = self.way1(src)
            output2 = self.way2(src)
            output3 = self.way3(src)
            output4 = self.way4(src)
            output5 = self.way5(src)
            output6 = self.way6(src)
            output7 = self.way7(src)
            output8 = self.way8(src)
            output9 = self.way9(src)
            output10 = self.way10(src)

            output = self.concat([output1, output2, output3, output4, output5, output6, output7, output8, output9, output10])

            output = self.dense1(output)
            output = self.dense2(output)
            output = self.dense3(output)
            output = self.dense4(output)
            output = self.dense5(output)
            output = self.dense6(output)

            return output

    with tf.device('/cpu:0'):
        model = KINT()
        model.build((None, 500, 5))
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.MeanSquaredError(), metrics=[TruePositivesM()])
        model.summary()
        model.load_weights("weights.03.hdf5")
        print("Making testing dataset...", flush=True)
        input, target = DataGen()
        counter = 0
        # print("MEMORY: {}".format(tf.config.experimental.get_memory_info('/cpu:0')))
        path = 'next_gen_simulation_usatest'
        file_list = sorted(os.listdir(path))
        for i in range(199):
            # print("Answer: {}".format(i))
            prediction = model(input[i].reshape((1, 500, 5))).numpy() ############# в переменной prediction хранится массив вероятностей
            # print(prediction)
            max_predicted = np.argmax(prediction.flatten())
            max_true = np.argmax(target[i])
            if abs(max_predicted - max_true) <= 1:
                counter += 1
            else:
                print(file_list[i], max_predicted)
                print(target[i])
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
