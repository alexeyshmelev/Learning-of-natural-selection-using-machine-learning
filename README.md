# Developing derictory
Link for training and testing data downloading: [Google Drive](https://drive.google.com/drive/folders/1x6GPoihKo1m2J6SnnCaCmT0-b3f2x0V9?usp=sharing)

# Start cluster shell

```
srun --pty --mincpus=2 --gpus=1 bash
```

# Compile SELAM

```
module load Python/Anaconda_v10.2019
module load libs/gsl/2.6
make -f Makefile
```

# Problems

* **Overfitting**

```
Model: "kint"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 500, 256)          25856     
_________________________________________________________________
conv1d_1 (Conv1D)            multiple                  0 (unused)
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 250, 256)          0         
_________________________________________________________________
batch_normalization (BatchNo (None, 500, 256)          1024      
_________________________________________________________________
activation (Activation)      (None, 500, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 250, 256)          1310976   
_________________________________________________________________
conv1d_3 (Conv1D)            multiple                  0 (unused)
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 125, 256)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 250, 256)          1024      
_________________________________________________________________
activation_1 (Activation)    (None, 250, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 125, 256)          1310976   
_________________________________________________________________
conv1d_5 (Conv1D)            multiple                  0 (unused)
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 25, 256)           0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 125, 256)          1024      
_________________________________________________________________
activation_2 (Activation)    (None, 125, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 25, 256)           1310976   
_________________________________________________________________
conv1d_7 (Conv1D)            multiple                  0 (unused)
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 5, 256)            0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 25, 256)           1024      
_________________________________________________________________
activation_3 (Activation)    (None, 25, 256)           0         
_________________________________________________________________
reshape (Reshape)            (None, 1280)              0         
_________________________________________________________________
dropout (Dropout)            multiple                  0 (unused)
_________________________________________________________________
dropout_1 (Dropout)          multiple                  0 (unused)
_________________________________________________________________
dropout_2 (Dropout)          multiple                  0 (unused)
_________________________________________________________________
dropout_3 (Dropout)          multiple                  0 (unused)
_________________________________________________________________
dense (Dense)                (None, 1000)              1281000   
_________________________________________________________________
dense_1 (Dense)              (None, 500)               500500    
_________________________________________________________________
dense_2 (Dense)              (None, 250)               125250    
_________________________________________________________________
dense_3 (Dense)              (None, 125)               31375     
_________________________________________________________________
dense_4 (Dense)              (None, 75)                9450      
_________________________________________________________________
dense_5 (Dense)              (None, 10)                760       
=================================================================
Total params: 5,911,215
Trainable params: 5,909,167
Non-trainable params: 2,048
_________________________________________________________________
Making testing dataset...
(199, 500, 5) (199, 1, 10)
Making training dataset...
(111993, 500, 5) (111993, 1, 10)
Training...
Epoch 1/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 104s 28ms/step - loss: 1.7144 - true_positives_m: 0.6565 - val_loss: 1.5885 - val_true_positives_m: 0.6823
Epoch 2/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.6474 - true_positives_m: 0.6866 - val_loss: 1.5804 - val_true_positives_m: 0.6927
Epoch 3/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.6330 - true_positives_m: 0.6936 - val_loss: 1.5741 - val_true_positives_m: 0.6979
Epoch 4/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.6202 - true_positives_m: 0.7000 - val_loss: 1.5404 - val_true_positives_m: 0.7188
Epoch 5/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.6087 - true_positives_m: 0.7063 - val_loss: 1.5620 - val_true_positives_m: 0.7240
Epoch 6/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.6006 - true_positives_m: 0.7113 - val_loss: 1.5106 - val_true_positives_m: 0.7396
Epoch 7/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5927 - true_positives_m: 0.7128 - val_loss: 1.5581 - val_true_positives_m: 0.7344
Epoch 8/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5853 - true_positives_m: 0.7136 - val_loss: 1.5145 - val_true_positives_m: 0.7448
Epoch 9/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5770 - true_positives_m: 0.7173 - val_loss: 1.5036 - val_true_positives_m: 0.7656
Epoch 10/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5696 - true_positives_m: 0.7204 - val_loss: 1.5276 - val_true_positives_m: 0.7604
Epoch 11/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 97s 28ms/step - loss: 1.5612 - true_positives_m: 0.7230 - val_loss: 1.5034 - val_true_positives_m: 0.7344
Epoch 12/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 98s 28ms/step - loss: 1.5520 - true_positives_m: 0.7257 - val_loss: 1.5106 - val_true_positives_m: 0.7500
Epoch 13/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5409 - true_positives_m: 0.7268 - val_loss: 1.5508 - val_true_positives_m: 0.7188
Epoch 14/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5300 - true_positives_m: 0.7295 - val_loss: 1.5222 - val_true_positives_m: 0.7708
Epoch 15/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5167 - true_positives_m: 0.7347 - val_loss: 1.5117 - val_true_positives_m: 0.7552
Epoch 16/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.5048 - true_positives_m: 0.7386 - val_loss: 1.5469 - val_true_positives_m: 0.7344
Epoch 17/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.4849 - true_positives_m: 0.7442 - val_loss: 1.5411 - val_true_positives_m: 0.7344
Epoch 18/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.4660 - true_positives_m: 0.7489 - val_loss: 1.5303 - val_true_positives_m: 0.7552
Epoch 19/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.4456 - true_positives_m: 0.7551 - val_loss: 1.5672 - val_true_positives_m: 0.7656
Epoch 20/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.4205 - true_positives_m: 0.7611 - val_loss: 1.6001 - val_true_positives_m: 0.7083
Epoch 21/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.3938 - true_positives_m: 0.7689 - val_loss: 1.6314 - val_true_positives_m: 0.7552
Epoch 22/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.3618 - true_positives_m: 0.7783 - val_loss: 1.6203 - val_true_positives_m: 0.7396
Epoch 23/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 29ms/step - loss: 1.3308 - true_positives_m: 0.7841 - val_loss: 1.5978 - val_true_positives_m: 0.7344
Epoch 24/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 28ms/step - loss: 1.2925 - true_positives_m: 0.7953 - val_loss: 1.6227 - val_true_positives_m: 0.7448
Epoch 25/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 29ms/step - loss: 1.2529 - true_positives_m: 0.8039 - val_loss: 1.6974 - val_true_positives_m: 0.7760
Epoch 26/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.2151 - true_positives_m: 0.8139 - val_loss: 1.7302 - val_true_positives_m: 0.7292
Epoch 27/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 28ms/step - loss: 1.1686 - true_positives_m: 0.8233 - val_loss: 1.8436 - val_true_positives_m: 0.7344
Epoch 28/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.1283 - true_positives_m: 0.8312 - val_loss: 1.7993 - val_true_positives_m: 0.7135
Epoch 29/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.0793 - true_positives_m: 0.8419 - val_loss: 1.8430 - val_true_positives_m: 0.6927
Epoch 30/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 1.0338 - true_positives_m: 0.8513 - val_loss: 1.9389 - val_true_positives_m: 0.7188
Epoch 31/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.9847 - true_positives_m: 0.8619 - val_loss: 1.9564 - val_true_positives_m: 0.7396
Epoch 32/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 29ms/step - loss: 0.9396 - true_positives_m: 0.8696 - val_loss: 2.0053 - val_true_positives_m: 0.7188
Epoch 33/100
LEARNING RATE: 1e-05
  25/3499 [..............................] - ETA: 1:38 - loss: 0.8198 - true_positives_m: 0.9000
Epoch 00033: saving model to /home/weights.33.hdf5
3499/3499 [==============================] - 100s 29ms/step - loss: 0.8943 - true_positives_m: 0.8767 - val_loss: 2.1271 - val_true_positives_m: 0.7448
Epoch 34/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 100s 29ms/step - loss: 0.8491 - true_positives_m: 0.8859 - val_loss: 2.2400 - val_true_positives_m: 0.7135
Epoch 35/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.8054 - true_positives_m: 0.8910 - val_loss: 2.1824 - val_true_positives_m: 0.7396
Epoch 36/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.7584 - true_positives_m: 0.8991 - val_loss: 2.3733 - val_true_positives_m: 0.7188
Epoch 37/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.7188 - true_positives_m: 0.9059 - val_loss: 2.3637 - val_true_positives_m: 0.7083
Epoch 38/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.6788 - true_positives_m: 0.9093 - val_loss: 2.4327 - val_true_positives_m: 0.6823
Epoch 39/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.6429 - true_positives_m: 0.9152 - val_loss: 2.5046 - val_true_positives_m: 0.7448
Epoch 40/100
LEARNING RATE: 1e-05
3499/3499 [==============================] - 99s 28ms/step - loss: 0.6021 - true_positives_m: 0.9208 - val_loss: 2.7631 - val_true_positives_m: 0.7135
Epoch 41/100
LEARNING RATE: 1e-05
 311/3499 [=>............................] - ETA: 1:29 - loss: 0.5281 - true_positives_m: 0.9342
Process finished with exit code -1

```

# Ensemble stats

'
```
C:\Users\Alex\AppData\Local\Programs\Python\Python39\python.exe C:/HSE/EPISTASIS/nn/main.py
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 3520207452200639013
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 6304890880
locality {
  bus_id: 1
  links {
  }
}
incarnation: 2479012375675079174
physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 2060 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5"
xla_global_id: 416903419
]
2.8.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 1, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(8527, 1001, 7) (8527, 1, 10)
Making testing dataset...
(834, 1001, 7) (834, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
265/266 [============================>.] - ETA: 0s - loss: 0.2611 - true_positives_m: 0.6988
Epoch 1: val_true_positives_m improved from -inf to 0.50481, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_1.hdf5
266/266 [==============================] - 11s 21ms/step - loss: 0.2610 - true_positives_m: 0.6994 - val_loss: 0.3504 - val_true_positives_m: 0.5048 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
266/266 [==============================] - ETA: 0s - loss: 0.2279 - true_positives_m: 0.7857
Epoch 2: val_true_positives_m improved from 0.50481 to 0.79688, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2279 - true_positives_m: 0.7857 - val_loss: 0.2232 - val_true_positives_m: 0.7969 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
266/266 [==============================] - ETA: 0s - loss: 0.2261 - true_positives_m: 0.7906
Epoch 3: val_true_positives_m improved from 0.79688 to 0.81010, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2261 - true_positives_m: 0.7906 - val_loss: 0.2177 - val_true_positives_m: 0.8101 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
263/266 [============================>.] - ETA: 0s - loss: 0.2255 - true_positives_m: 0.7917
Epoch 4: val_true_positives_m improved from 0.81010 to 0.83894, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2254 - true_positives_m: 0.7922 - val_loss: 0.2155 - val_true_positives_m: 0.8389 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
263/266 [============================>.] - ETA: 0s - loss: 0.2227 - true_positives_m: 0.7974
Epoch 5: val_true_positives_m improved from 0.83894 to 0.84856, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2228 - true_positives_m: 0.7979 - val_loss: 0.2157 - val_true_positives_m: 0.8486 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
265/266 [============================>.] - ETA: 0s - loss: 0.2218 - true_positives_m: 0.7967
Epoch 6: val_true_positives_m did not improve from 0.84856
266/266 [==============================] - 5s 17ms/step - loss: 0.2218 - true_positives_m: 0.7969 - val_loss: 0.2152 - val_true_positives_m: 0.8269 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
265/266 [============================>.] - ETA: 0s - loss: 0.2205 - true_positives_m: 0.7960
Epoch 7: val_true_positives_m did not improve from 0.84856
266/266 [==============================] - 5s 17ms/step - loss: 0.2205 - true_positives_m: 0.7956 - val_loss: 0.2160 - val_true_positives_m: 0.8450 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
263/266 [============================>.] - ETA: 0s - loss: 0.2205 - true_positives_m: 0.8004
Epoch 8: val_true_positives_m did not improve from 0.84856
266/266 [==============================] - 5s 17ms/step - loss: 0.2205 - true_positives_m: 0.8004 - val_loss: 0.2154 - val_true_positives_m: 0.8281 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
266/266 [==============================] - ETA: 0s - loss: 0.2181 - true_positives_m: 0.8017
Epoch 9: val_true_positives_m did not improve from 0.84856
266/266 [==============================] - 5s 17ms/step - loss: 0.2181 - true_positives_m: 0.8017 - val_loss: 0.2226 - val_true_positives_m: 0.7861 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
265/266 [============================>.] - ETA: 0s - loss: 0.2199 - true_positives_m: 0.7959
Epoch 10: val_true_positives_m did not improve from 0.84856
266/266 [==============================] - 5s 17ms/step - loss: 0.2199 - true_positives_m: 0.7959 - val_loss: 0.2162 - val_true_positives_m: 0.8353 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 1, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(8527, 1001, 7) (8527, 1, 10)
Making testing dataset...
(834, 1001, 7) (834, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
266/266 [==============================] - ETA: 0s - loss: 0.2605 - true_positives_m: 0.7064
Epoch 1: val_true_positives_m improved from -inf to 0.56250, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_1.hdf5
266/266 [==============================] - 10s 21ms/step - loss: 0.2605 - true_positives_m: 0.7064 - val_loss: 0.3022 - val_true_positives_m: 0.5625 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
263/266 [============================>.] - ETA: 0s - loss: 0.2276 - true_positives_m: 0.7902
Epoch 2: val_true_positives_m improved from 0.56250 to 0.82572, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2276 - true_positives_m: 0.7902 - val_loss: 0.2159 - val_true_positives_m: 0.8257 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
266/266 [==============================] - ETA: 0s - loss: 0.2258 - true_positives_m: 0.7917
Epoch 3: val_true_positives_m did not improve from 0.82572
266/266 [==============================] - 5s 17ms/step - loss: 0.2258 - true_positives_m: 0.7917 - val_loss: 0.2144 - val_true_positives_m: 0.8233 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
266/266 [==============================] - ETA: 0s - loss: 0.2250 - true_positives_m: 0.7867
Epoch 4: val_true_positives_m did not improve from 0.82572
266/266 [==============================] - 5s 17ms/step - loss: 0.2250 - true_positives_m: 0.7867 - val_loss: 0.2182 - val_true_positives_m: 0.8233 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
264/266 [============================>.] - ETA: 0s - loss: 0.2233 - true_positives_m: 0.7934
Epoch 5: val_true_positives_m did not improve from 0.82572
266/266 [==============================] - 5s 17ms/step - loss: 0.2233 - true_positives_m: 0.7932 - val_loss: 0.2175 - val_true_positives_m: 0.8161 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
266/266 [==============================] - ETA: 0s - loss: 0.2225 - true_positives_m: 0.7983
Epoch 6: val_true_positives_m did not improve from 0.82572
266/266 [==============================] - 5s 17ms/step - loss: 0.2225 - true_positives_m: 0.7983 - val_loss: 0.2182 - val_true_positives_m: 0.8065 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
264/266 [============================>.] - ETA: 0s - loss: 0.2202 - true_positives_m: 0.7943
Epoch 7: val_true_positives_m improved from 0.82572 to 0.83413, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2202 - true_positives_m: 0.7945 - val_loss: 0.2132 - val_true_positives_m: 0.8341 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
265/266 [============================>.] - ETA: 0s - loss: 0.2164 - true_positives_m: 0.8104
Epoch 8: val_true_positives_m improved from 0.83413 to 0.83774, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_1.hdf5
266/266 [==============================] - 5s 17ms/step - loss: 0.2165 - true_positives_m: 0.8103 - val_loss: 0.2127 - val_true_positives_m: 0.8377 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
264/266 [============================>.] - ETA: 0s - loss: 0.2156 - true_positives_m: 0.8033
Epoch 9: val_true_positives_m improved from 0.83774 to 0.83894, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_1.hdf5
266/266 [==============================] - 5s 18ms/step - loss: 0.2156 - true_positives_m: 0.8027 - val_loss: 0.2157 - val_true_positives_m: 0.8389 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
265/266 [============================>.] - ETA: 0s - loss: 0.2133 - true_positives_m: 0.8103
Epoch 10: val_true_positives_m did not improve from 0.83894
266/266 [==============================] - 4s 17ms/step - loss: 0.2133 - true_positives_m: 0.8103 - val_loss: 0.2189 - val_true_positives_m: 0.8005 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 2, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10008, 1001, 7) (10008, 1, 10)
Making testing dataset...
(952, 1001, 7) (952, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
309/312 [============================>.] - ETA: 0s - loss: 0.2599 - true_positives_m: 0.7125
Epoch 1: val_true_positives_m improved from -inf to 0.31789, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_2.hdf5
312/312 [==============================] - 11s 20ms/step - loss: 0.2598 - true_positives_m: 0.7122 - val_loss: 0.4911 - val_true_positives_m: 0.3179 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
310/312 [============================>.] - ETA: 0s - loss: 0.2266 - true_positives_m: 0.7925
Epoch 2: val_true_positives_m improved from 0.31789 to 0.79634, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_2.hdf5
312/312 [==============================] - 5s 17ms/step - loss: 0.2265 - true_positives_m: 0.7932 - val_loss: 0.2224 - val_true_positives_m: 0.7963 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
310/312 [============================>.] - ETA: 0s - loss: 0.2248 - true_positives_m: 0.7962
Epoch 3: val_true_positives_m improved from 0.79634 to 0.79849, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_2.hdf5
312/312 [==============================] - 5s 18ms/step - loss: 0.2249 - true_positives_m: 0.7958 - val_loss: 0.2207 - val_true_positives_m: 0.7985 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
312/312 [==============================] - ETA: 0s - loss: 0.2229 - true_positives_m: 0.7976
Epoch 4: val_true_positives_m improved from 0.79849 to 0.80927, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_2.hdf5
312/312 [==============================] - 5s 18ms/step - loss: 0.2229 - true_positives_m: 0.7976 - val_loss: 0.2178 - val_true_positives_m: 0.8093 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
312/312 [==============================] - ETA: 0s - loss: 0.2234 - true_positives_m: 0.7914
Epoch 5: val_true_positives_m did not improve from 0.80927
312/312 [==============================] - 5s 17ms/step - loss: 0.2234 - true_positives_m: 0.7914 - val_loss: 0.2190 - val_true_positives_m: 0.8082 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
310/312 [============================>.] - ETA: 0s - loss: 0.2193 - true_positives_m: 0.8042
Epoch 6: val_true_positives_m did not improve from 0.80927
312/312 [==============================] - 5s 17ms/step - loss: 0.2193 - true_positives_m: 0.8043 - val_loss: 0.2237 - val_true_positives_m: 0.7823 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
311/312 [============================>.] - ETA: 0s - loss: 0.2194 - true_positives_m: 0.7961
Epoch 7: val_true_positives_m improved from 0.80927 to 0.81142, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_2.hdf5
312/312 [==============================] - 5s 17ms/step - loss: 0.2194 - true_positives_m: 0.7961 - val_loss: 0.2167 - val_true_positives_m: 0.8114 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
312/312 [==============================] - ETA: 0s - loss: 0.2173 - true_positives_m: 0.8015
Epoch 8: val_true_positives_m did not improve from 0.81142
312/312 [==============================] - 5s 17ms/step - loss: 0.2173 - true_positives_m: 0.8015 - val_loss: 0.2175 - val_true_positives_m: 0.8103 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
311/312 [============================>.] - ETA: 0s - loss: 0.2160 - true_positives_m: 0.8035
Epoch 9: val_true_positives_m did not improve from 0.81142
312/312 [==============================] - 5s 17ms/step - loss: 0.2160 - true_positives_m: 0.8030 - val_loss: 0.2218 - val_true_positives_m: 0.8039 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
311/312 [============================>.] - ETA: 0s - loss: 0.2134 - true_positives_m: 0.8066
Epoch 10: val_true_positives_m did not improve from 0.81142
312/312 [==============================] - 5s 17ms/step - loss: 0.2134 - true_positives_m: 0.8065 - val_loss: 0.2239 - val_true_positives_m: 0.7953 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 2, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10008, 1001, 7) (10008, 1, 10)
Making testing dataset...
(952, 1001, 7) (952, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
309/312 [============================>.] - ETA: 0s - loss: 0.2553 - true_positives_m: 0.7140
Epoch 1: val_true_positives_m improved from -inf to 0.67457, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_2.hdf5
312/312 [==============================] - 11s 21ms/step - loss: 0.2550 - true_positives_m: 0.7147 - val_loss: 0.2643 - val_true_positives_m: 0.6746 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
309/312 [============================>.] - ETA: 0s - loss: 0.2269 - true_positives_m: 0.7957
Epoch 2: val_true_positives_m improved from 0.67457 to 0.83513, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_2.hdf5
312/312 [==============================] - 6s 18ms/step - loss: 0.2268 - true_positives_m: 0.7962 - val_loss: 0.2185 - val_true_positives_m: 0.8351 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
311/312 [============================>.] - ETA: 0s - loss: 0.2242 - true_positives_m: 0.8063
Epoch 3: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 17ms/step - loss: 0.2242 - true_positives_m: 0.8065 - val_loss: 0.2185 - val_true_positives_m: 0.8222 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
310/312 [============================>.] - ETA: 0s - loss: 0.2216 - true_positives_m: 0.8047
Epoch 4: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 17ms/step - loss: 0.2216 - true_positives_m: 0.8048 - val_loss: 0.2171 - val_true_positives_m: 0.8276 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
309/312 [============================>.] - ETA: 0s - loss: 0.2210 - true_positives_m: 0.8055
Epoch 5: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 18ms/step - loss: 0.2211 - true_positives_m: 0.8056 - val_loss: 0.2240 - val_true_positives_m: 0.8082 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
310/312 [============================>.] - ETA: 0s - loss: 0.2207 - true_positives_m: 0.8116
Epoch 6: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 17ms/step - loss: 0.2207 - true_positives_m: 0.8117 - val_loss: 0.2181 - val_true_positives_m: 0.8341 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
311/312 [============================>.] - ETA: 0s - loss: 0.2181 - true_positives_m: 0.8134
Epoch 7: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 18ms/step - loss: 0.2182 - true_positives_m: 0.8131 - val_loss: 0.2163 - val_true_positives_m: 0.8276 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
310/312 [============================>.] - ETA: 0s - loss: 0.2169 - true_positives_m: 0.8120
Epoch 8: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 18ms/step - loss: 0.2168 - true_positives_m: 0.8125 - val_loss: 0.2220 - val_true_positives_m: 0.8157 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
311/312 [============================>.] - ETA: 0s - loss: 0.2146 - true_positives_m: 0.8135
Epoch 9: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 18ms/step - loss: 0.2146 - true_positives_m: 0.8135 - val_loss: 0.2163 - val_true_positives_m: 0.8265 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
311/312 [============================>.] - ETA: 0s - loss: 0.2125 - true_positives_m: 0.8180
Epoch 10: val_true_positives_m did not improve from 0.83513
312/312 [==============================] - 5s 17ms/step - loss: 0.2124 - true_positives_m: 0.8182 - val_loss: 0.2170 - val_true_positives_m: 0.8190 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 3, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9130, 1001, 7) (9130, 1, 10)
Making testing dataset...
(894, 1001, 7) (894, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
282/285 [============================>.] - ETA: 0s - loss: 0.2631 - true_positives_m: 0.6845
Epoch 1: val_true_positives_m improved from -inf to 0.68634, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 11s 21ms/step - loss: 0.2631 - true_positives_m: 0.6846 - val_loss: 0.2495 - val_true_positives_m: 0.6863 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
283/285 [============================>.] - ETA: 0s - loss: 0.2322 - true_positives_m: 0.7656
Epoch 2: val_true_positives_m improved from 0.68634 to 0.81250, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 5s 18ms/step - loss: 0.2322 - true_positives_m: 0.7658 - val_loss: 0.2240 - val_true_positives_m: 0.8125 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
284/285 [============================>.] - ETA: 0s - loss: 0.2304 - true_positives_m: 0.7772
Epoch 3: val_true_positives_m did not improve from 0.81250
285/285 [==============================] - 5s 17ms/step - loss: 0.2303 - true_positives_m: 0.7771 - val_loss: 0.2276 - val_true_positives_m: 0.7940 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
285/285 [==============================] - ETA: 0s - loss: 0.2274 - true_positives_m: 0.7827
Epoch 4: val_true_positives_m improved from 0.81250 to 0.81366, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 5s 17ms/step - loss: 0.2274 - true_positives_m: 0.7827 - val_loss: 0.2235 - val_true_positives_m: 0.8137 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
285/285 [==============================] - ETA: 0s - loss: 0.2262 - true_positives_m: 0.7854
Epoch 5: val_true_positives_m did not improve from 0.81366
285/285 [==============================] - 5s 17ms/step - loss: 0.2262 - true_positives_m: 0.7854 - val_loss: 0.2224 - val_true_positives_m: 0.8056 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
285/285 [==============================] - ETA: 0s - loss: 0.2251 - true_positives_m: 0.7864
Epoch 6: val_true_positives_m did not improve from 0.81366
285/285 [==============================] - 5s 17ms/step - loss: 0.2251 - true_positives_m: 0.7864 - val_loss: 0.2221 - val_true_positives_m: 0.8067 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
283/285 [============================>.] - ETA: 0s - loss: 0.2214 - true_positives_m: 0.7949
Epoch 7: val_true_positives_m did not improve from 0.81366
285/285 [==============================] - 5s 17ms/step - loss: 0.2215 - true_positives_m: 0.7951 - val_loss: 0.2219 - val_true_positives_m: 0.8113 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
282/285 [============================>.] - ETA: 0s - loss: 0.2232 - true_positives_m: 0.7904
Epoch 8: val_true_positives_m improved from 0.81366 to 0.81481, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 5s 18ms/step - loss: 0.2231 - true_positives_m: 0.7908 - val_loss: 0.2237 - val_true_positives_m: 0.8148 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
284/285 [============================>.] - ETA: 0s - loss: 0.2204 - true_positives_m: 0.7936
Epoch 9: val_true_positives_m improved from 0.81481 to 0.81597, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 5s 18ms/step - loss: 0.2204 - true_positives_m: 0.7936 - val_loss: 0.2218 - val_true_positives_m: 0.8160 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
283/285 [============================>.] - ETA: 0s - loss: 0.2192 - true_positives_m: 0.7889
Epoch 10: val_true_positives_m improved from 0.81597 to 0.82639, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_3.hdf5
285/285 [==============================] - 5s 17ms/step - loss: 0.2192 - true_positives_m: 0.7893 - val_loss: 0.2212 - val_true_positives_m: 0.8264 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 3, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9130, 1001, 7) (9130, 1, 10)
Making testing dataset...
(894, 1001, 7) (894, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
285/285 [==============================] - ETA: 0s - loss: 0.2658 - true_positives_m: 0.6782
Epoch 1: val_true_positives_m improved from -inf to 0.33796, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_3.hdf5
285/285 [==============================] - 11s 21ms/step - loss: 0.2658 - true_positives_m: 0.6782 - val_loss: 0.4370 - val_true_positives_m: 0.3380 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
284/285 [============================>.] - ETA: 0s - loss: 0.2312 - true_positives_m: 0.7730
Epoch 2: val_true_positives_m improved from 0.33796 to 0.82639, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_3.hdf5
285/285 [==============================] - 5s 18ms/step - loss: 0.2312 - true_positives_m: 0.7732 - val_loss: 0.2248 - val_true_positives_m: 0.8264 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
284/285 [============================>.] - ETA: 0s - loss: 0.2272 - true_positives_m: 0.7840
Epoch 3: val_true_positives_m did not improve from 0.82639
285/285 [==============================] - 5s 17ms/step - loss: 0.2272 - true_positives_m: 0.7842 - val_loss: 0.2217 - val_true_positives_m: 0.8264 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
282/285 [============================>.] - ETA: 0s - loss: 0.2250 - true_positives_m: 0.7900
Epoch 4: val_true_positives_m improved from 0.82639 to 0.83565, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_3.hdf5
285/285 [==============================] - 5s 17ms/step - loss: 0.2249 - true_positives_m: 0.7906 - val_loss: 0.2203 - val_true_positives_m: 0.8356 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
283/285 [============================>.] - ETA: 0s - loss: 0.2236 - true_positives_m: 0.7916
Epoch 5: val_true_positives_m did not improve from 0.83565
285/285 [==============================] - 5s 17ms/step - loss: 0.2237 - true_positives_m: 0.7910 - val_loss: 0.2224 - val_true_positives_m: 0.8125 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
282/285 [============================>.] - ETA: 0s - loss: 0.2209 - true_positives_m: 0.7950
Epoch 6: val_true_positives_m improved from 0.83565 to 0.84491, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_3.hdf5
285/285 [==============================] - 5s 17ms/step - loss: 0.2211 - true_positives_m: 0.7945 - val_loss: 0.2205 - val_true_positives_m: 0.8449 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
285/285 [==============================] - ETA: 0s - loss: 0.2192 - true_positives_m: 0.8011
Epoch 7: val_true_positives_m did not improve from 0.84491
285/285 [==============================] - 5s 17ms/step - loss: 0.2192 - true_positives_m: 0.8011 - val_loss: 0.2224 - val_true_positives_m: 0.8356 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
282/285 [============================>.] - ETA: 0s - loss: 0.2170 - true_positives_m: 0.8006
Epoch 8: val_true_positives_m did not improve from 0.84491
285/285 [==============================] - 5s 17ms/step - loss: 0.2170 - true_positives_m: 0.8007 - val_loss: 0.2244 - val_true_positives_m: 0.8171 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
283/285 [============================>.] - ETA: 0s - loss: 0.2161 - true_positives_m: 0.7951
Epoch 9: val_true_positives_m did not improve from 0.84491
285/285 [==============================] - 5s 17ms/step - loss: 0.2162 - true_positives_m: 0.7951 - val_loss: 0.2236 - val_true_positives_m: 0.8403 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
285/285 [==============================] - ETA: 0s - loss: 0.2113 - true_positives_m: 0.8105
Epoch 10: val_true_positives_m did not improve from 0.84491
285/285 [==============================] - 5s 17ms/step - loss: 0.2113 - true_positives_m: 0.8105 - val_loss: 0.2272 - val_true_positives_m: 0.7986 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 4, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9963, 1001, 7) (9963, 1, 10)
Making testing dataset...
(957, 1001, 7) (957, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
310/311 [============================>.] - ETA: 0s - loss: 0.2644 - true_positives_m: 0.6681
Epoch 1: val_true_positives_m improved from -inf to 0.56142, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_4.hdf5
311/311 [==============================] - 11s 20ms/step - loss: 0.2644 - true_positives_m: 0.6682 - val_loss: 0.2932 - val_true_positives_m: 0.5614 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
308/311 [============================>.] - ETA: 0s - loss: 0.2345 - true_positives_m: 0.7585
Epoch 2: val_true_positives_m improved from 0.56142 to 0.77802, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_4.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2346 - true_positives_m: 0.7580 - val_loss: 0.2314 - val_true_positives_m: 0.7780 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
311/311 [==============================] - ETA: 0s - loss: 0.2313 - true_positives_m: 0.7650
Epoch 3: val_true_positives_m improved from 0.77802 to 0.78233, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_4.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2313 - true_positives_m: 0.7650 - val_loss: 0.2325 - val_true_positives_m: 0.7823 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
310/311 [============================>.] - ETA: 0s - loss: 0.2288 - true_positives_m: 0.7709
Epoch 4: val_true_positives_m improved from 0.78233 to 0.79634, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_4.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2287 - true_positives_m: 0.7711 - val_loss: 0.2284 - val_true_positives_m: 0.7963 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
310/311 [============================>.] - ETA: 0s - loss: 0.2265 - true_positives_m: 0.7808
Epoch 5: val_true_positives_m did not improve from 0.79634
311/311 [==============================] - 5s 17ms/step - loss: 0.2265 - true_positives_m: 0.7806 - val_loss: 0.2295 - val_true_positives_m: 0.7866 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
311/311 [==============================] - ETA: 0s - loss: 0.2252 - true_positives_m: 0.7829
Epoch 6: val_true_positives_m improved from 0.79634 to 0.79957, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_4.hdf5
311/311 [==============================] - 6s 18ms/step - loss: 0.2252 - true_positives_m: 0.7829 - val_loss: 0.2293 - val_true_positives_m: 0.7996 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
311/311 [==============================] - ETA: 0s - loss: 0.2229 - true_positives_m: 0.7847
Epoch 7: val_true_positives_m did not improve from 0.79957
311/311 [==============================] - 5s 17ms/step - loss: 0.2229 - true_positives_m: 0.7847 - val_loss: 0.2256 - val_true_positives_m: 0.7899 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
309/311 [============================>.] - ETA: 0s - loss: 0.2203 - true_positives_m: 0.7872
Epoch 8: val_true_positives_m did not improve from 0.79957
311/311 [==============================] - 5s 17ms/step - loss: 0.2202 - true_positives_m: 0.7874 - val_loss: 0.2294 - val_true_positives_m: 0.7823 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
311/311 [==============================] - ETA: 0s - loss: 0.2175 - true_positives_m: 0.7891
Epoch 9: val_true_positives_m did not improve from 0.79957
311/311 [==============================] - 5s 17ms/step - loss: 0.2175 - true_positives_m: 0.7891 - val_loss: 0.2299 - val_true_positives_m: 0.7877 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
311/311 [==============================] - ETA: 0s - loss: 0.2156 - true_positives_m: 0.7866
Epoch 10: val_true_positives_m did not improve from 0.79957
311/311 [==============================] - 5s 17ms/step - loss: 0.2156 - true_positives_m: 0.7866 - val_loss: 0.2345 - val_true_positives_m: 0.7963 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 4, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9963, 1001, 7) (9963, 1, 10)
Making testing dataset...
(957, 1001, 7) (957, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
309/311 [============================>.] - ETA: 0s - loss: 0.2639 - true_positives_m: 0.6713
Epoch 1: val_true_positives_m improved from -inf to 0.45151, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_4.hdf5
311/311 [==============================] - 11s 20ms/step - loss: 0.2638 - true_positives_m: 0.6714 - val_loss: 0.3345 - val_true_positives_m: 0.4515 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
311/311 [==============================] - ETA: 0s - loss: 0.2326 - true_positives_m: 0.7587
Epoch 2: val_true_positives_m improved from 0.45151 to 0.78772, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_4.hdf5
311/311 [==============================] - 6s 18ms/step - loss: 0.2326 - true_positives_m: 0.7587 - val_loss: 0.2281 - val_true_positives_m: 0.7877 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
311/311 [==============================] - ETA: 0s - loss: 0.2303 - true_positives_m: 0.7685
Epoch 3: val_true_positives_m improved from 0.78772 to 0.80388, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_4.hdf5
311/311 [==============================] - 5s 18ms/step - loss: 0.2303 - true_positives_m: 0.7685 - val_loss: 0.2304 - val_true_positives_m: 0.8039 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
308/311 [============================>.] - ETA: 0s - loss: 0.2295 - true_positives_m: 0.7777
Epoch 4: val_true_positives_m did not improve from 0.80388
311/311 [==============================] - 5s 17ms/step - loss: 0.2294 - true_positives_m: 0.7779 - val_loss: 0.2255 - val_true_positives_m: 0.7866 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
308/311 [============================>.] - ETA: 0s - loss: 0.2278 - true_positives_m: 0.7702
Epoch 5: val_true_positives_m did not improve from 0.80388
311/311 [==============================] - 5s 17ms/step - loss: 0.2278 - true_positives_m: 0.7709 - val_loss: 0.2250 - val_true_positives_m: 0.8017 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
310/311 [============================>.] - ETA: 0s - loss: 0.2262 - true_positives_m: 0.7834
Epoch 6: val_true_positives_m did not improve from 0.80388
311/311 [==============================] - 5s 17ms/step - loss: 0.2262 - true_positives_m: 0.7834 - val_loss: 0.2265 - val_true_positives_m: 0.7877 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
310/311 [============================>.] - ETA: 0s - loss: 0.2249 - true_positives_m: 0.7792
Epoch 7: val_true_positives_m did not improve from 0.80388
311/311 [==============================] - 5s 17ms/step - loss: 0.2248 - true_positives_m: 0.7794 - val_loss: 0.2257 - val_true_positives_m: 0.7856 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
308/311 [============================>.] - ETA: 0s - loss: 0.2223 - true_positives_m: 0.7853
Epoch 8: val_true_positives_m improved from 0.80388 to 0.81142, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_4.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2222 - true_positives_m: 0.7857 - val_loss: 0.2283 - val_true_positives_m: 0.8114 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
311/311 [==============================] - ETA: 0s - loss: 0.2189 - true_positives_m: 0.7945
Epoch 9: val_true_positives_m did not improve from 0.81142
311/311 [==============================] - 5s 17ms/step - loss: 0.2189 - true_positives_m: 0.7945 - val_loss: 0.2306 - val_true_positives_m: 0.7791 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
308/311 [============================>.] - ETA: 0s - loss: 0.2168 - true_positives_m: 0.7996
Epoch 10: val_true_positives_m did not improve from 0.81142
311/311 [==============================] - 5s 17ms/step - loss: 0.2168 - true_positives_m: 0.7995 - val_loss: 0.2270 - val_true_positives_m: 0.7963 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 5, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9667, 1001, 7) (9667, 1, 10)
Making testing dataset...
(1018, 1001, 7) (1018, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
302/302 [==============================] - ETA: 0s - loss: 0.2718 - true_positives_m: 0.6319
Epoch 1: val_true_positives_m improved from -inf to 0.43145, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_5.hdf5
302/302 [==============================] - 11s 21ms/step - loss: 0.2718 - true_positives_m: 0.6319 - val_loss: 0.3163 - val_true_positives_m: 0.4315 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
302/302 [==============================] - ETA: 0s - loss: 0.2409 - true_positives_m: 0.7263
Epoch 2: val_true_positives_m improved from 0.43145 to 0.72782, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_5.hdf5
302/302 [==============================] - 5s 18ms/step - loss: 0.2409 - true_positives_m: 0.7263 - val_loss: 0.2446 - val_true_positives_m: 0.7278 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
300/302 [============================>.] - ETA: 0s - loss: 0.2387 - true_positives_m: 0.7349
Epoch 3: val_true_positives_m improved from 0.72782 to 0.74698, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_5.hdf5
302/302 [==============================] - 6s 18ms/step - loss: 0.2387 - true_positives_m: 0.7347 - val_loss: 0.2396 - val_true_positives_m: 0.7470 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
300/302 [============================>.] - ETA: 0s - loss: 0.2371 - true_positives_m: 0.7361
Epoch 4: val_true_positives_m did not improve from 0.74698
302/302 [==============================] - 5s 17ms/step - loss: 0.2370 - true_positives_m: 0.7367 - val_loss: 0.2380 - val_true_positives_m: 0.7429 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
299/302 [============================>.] - ETA: 0s - loss: 0.2347 - true_positives_m: 0.7416
Epoch 5: val_true_positives_m did not improve from 0.74698
302/302 [==============================] - 5s 17ms/step - loss: 0.2348 - true_positives_m: 0.7418 - val_loss: 0.2402 - val_true_positives_m: 0.7409 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
299/302 [============================>.] - ETA: 0s - loss: 0.2331 - true_positives_m: 0.7474
Epoch 6: val_true_positives_m improved from 0.74698 to 0.75605, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_5.hdf5
302/302 [==============================] - 5s 17ms/step - loss: 0.2330 - true_positives_m: 0.7475 - val_loss: 0.2380 - val_true_positives_m: 0.7560 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
302/302 [==============================] - ETA: 0s - loss: 0.2318 - true_positives_m: 0.7495
Epoch 7: val_true_positives_m improved from 0.75605 to 0.76512, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_5.hdf5
302/302 [==============================] - 5s 18ms/step - loss: 0.2318 - true_positives_m: 0.7495 - val_loss: 0.2386 - val_true_positives_m: 0.7651 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
300/302 [============================>.] - ETA: 0s - loss: 0.2293 - true_positives_m: 0.7574
Epoch 8: val_true_positives_m did not improve from 0.76512
302/302 [==============================] - 5s 17ms/step - loss: 0.2295 - true_positives_m: 0.7570 - val_loss: 0.2380 - val_true_positives_m: 0.7429 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
300/302 [============================>.] - ETA: 0s - loss: 0.2270 - true_positives_m: 0.7578
Epoch 9: val_true_positives_m did not improve from 0.76512
302/302 [==============================] - 5s 17ms/step - loss: 0.2269 - true_positives_m: 0.7581 - val_loss: 0.2439 - val_true_positives_m: 0.7470 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
299/302 [============================>.] - ETA: 0s - loss: 0.2250 - true_positives_m: 0.7619
Epoch 10: val_true_positives_m did not improve from 0.76512
302/302 [==============================] - 5s 17ms/step - loss: 0.2249 - true_positives_m: 0.7619 - val_loss: 0.2423 - val_true_positives_m: 0.7510 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 5, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9667, 1001, 7) (9667, 1, 10)
Making testing dataset...
(1018, 1001, 7) (1018, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
302/302 [==============================] - ETA: 0s - loss: 0.2702 - true_positives_m: 0.6499
Epoch 1: val_true_positives_m improved from -inf to 0.62903, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_5.hdf5
302/302 [==============================] - 11s 21ms/step - loss: 0.2702 - true_positives_m: 0.6499 - val_loss: 0.2693 - val_true_positives_m: 0.6290 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
299/302 [============================>.] - ETA: 0s - loss: 0.2403 - true_positives_m: 0.7322
Epoch 2: val_true_positives_m improved from 0.62903 to 0.73790, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_5.hdf5
302/302 [==============================] - 5s 18ms/step - loss: 0.2402 - true_positives_m: 0.7328 - val_loss: 0.2384 - val_true_positives_m: 0.7379 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
301/302 [============================>.] - ETA: 0s - loss: 0.2372 - true_positives_m: 0.7415
Epoch 3: val_true_positives_m did not improve from 0.73790
302/302 [==============================] - 5s 18ms/step - loss: 0.2373 - true_positives_m: 0.7414 - val_loss: 0.2384 - val_true_positives_m: 0.7208 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
299/302 [============================>.] - ETA: 0s - loss: 0.2349 - true_positives_m: 0.7491
Epoch 4: val_true_positives_m improved from 0.73790 to 0.75706, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_5.hdf5
302/302 [==============================] - 5s 18ms/step - loss: 0.2350 - true_positives_m: 0.7487 - val_loss: 0.2390 - val_true_positives_m: 0.7571 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
300/302 [============================>.] - ETA: 0s - loss: 0.2339 - true_positives_m: 0.7462
Epoch 5: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 18ms/step - loss: 0.2340 - true_positives_m: 0.7457 - val_loss: 0.2368 - val_true_positives_m: 0.7540 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
301/302 [============================>.] - ETA: 0s - loss: 0.2306 - true_positives_m: 0.7579
Epoch 6: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 18ms/step - loss: 0.2307 - true_positives_m: 0.7581 - val_loss: 0.2375 - val_true_positives_m: 0.7440 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
301/302 [============================>.] - ETA: 0s - loss: 0.2296 - true_positives_m: 0.7580
Epoch 7: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 18ms/step - loss: 0.2299 - true_positives_m: 0.7575 - val_loss: 0.2368 - val_true_positives_m: 0.7349 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
301/302 [============================>.] - ETA: 0s - loss: 0.2263 - true_positives_m: 0.7632
Epoch 8: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 17ms/step - loss: 0.2264 - true_positives_m: 0.7625 - val_loss: 0.2422 - val_true_positives_m: 0.7167 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
301/302 [============================>.] - ETA: 0s - loss: 0.2248 - true_positives_m: 0.7633
Epoch 9: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 18ms/step - loss: 0.2248 - true_positives_m: 0.7635 - val_loss: 0.2420 - val_true_positives_m: 0.7369 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
300/302 [============================>.] - ETA: 0s - loss: 0.2219 - true_positives_m: 0.7708
Epoch 10: val_true_positives_m did not improve from 0.75706
302/302 [==============================] - 5s 18ms/step - loss: 0.2219 - true_positives_m: 0.7710 - val_loss: 0.2472 - val_true_positives_m: 0.7026 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 6, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9706, 1001, 7) (9706, 1, 10)
Making testing dataset...
(965, 1001, 7) (965, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
303/303 [==============================] - ETA: 0s - loss: 0.2772 - true_positives_m: 0.6094
Epoch 1: val_true_positives_m improved from -inf to 0.22396, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_6.hdf5
303/303 [==============================] - 12s 21ms/step - loss: 0.2772 - true_positives_m: 0.6094 - val_loss: 0.5479 - val_true_positives_m: 0.2240 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
301/303 [============================>.] - ETA: 0s - loss: 0.2496 - true_positives_m: 0.6918
Epoch 2: val_true_positives_m improved from 0.22396 to 0.69583, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_6.hdf5
303/303 [==============================] - 5s 18ms/step - loss: 0.2495 - true_positives_m: 0.6926 - val_loss: 0.2523 - val_true_positives_m: 0.6958 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
301/303 [============================>.] - ETA: 0s - loss: 0.2457 - true_positives_m: 0.7039
Epoch 3: val_true_positives_m did not improve from 0.69583
303/303 [==============================] - 5s 17ms/step - loss: 0.2456 - true_positives_m: 0.7045 - val_loss: 0.2659 - val_true_positives_m: 0.6187 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
300/303 [============================>.] - ETA: 0s - loss: 0.2442 - true_positives_m: 0.7119
Epoch 4: val_true_positives_m improved from 0.69583 to 0.73646, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_6.hdf5
303/303 [==============================] - 5s 17ms/step - loss: 0.2441 - true_positives_m: 0.7124 - val_loss: 0.2422 - val_true_positives_m: 0.7365 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
300/303 [============================>.] - ETA: 0s - loss: 0.2431 - true_positives_m: 0.7168
Epoch 5: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2431 - true_positives_m: 0.7168 - val_loss: 0.2434 - val_true_positives_m: 0.7208 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
300/303 [============================>.] - ETA: 0s - loss: 0.2404 - true_positives_m: 0.7251
Epoch 6: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2404 - true_positives_m: 0.7245 - val_loss: 0.2433 - val_true_positives_m: 0.7312 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
303/303 [==============================] - ETA: 0s - loss: 0.2391 - true_positives_m: 0.7215
Epoch 7: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2391 - true_positives_m: 0.7215 - val_loss: 0.2444 - val_true_positives_m: 0.7250 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
303/303 [==============================] - ETA: 0s - loss: 0.2370 - true_positives_m: 0.7248
Epoch 8: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2370 - true_positives_m: 0.7248 - val_loss: 0.2488 - val_true_positives_m: 0.6917 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
300/303 [============================>.] - ETA: 0s - loss: 0.2339 - true_positives_m: 0.7306
Epoch 9: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2339 - true_positives_m: 0.7304 - val_loss: 0.2488 - val_true_positives_m: 0.7167 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
302/303 [============================>.] - ETA: 0s - loss: 0.2318 - true_positives_m: 0.7345
Epoch 10: val_true_positives_m did not improve from 0.73646
303/303 [==============================] - 5s 17ms/step - loss: 0.2318 - true_positives_m: 0.7347 - val_loss: 0.2436 - val_true_positives_m: 0.7208 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 6, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9706, 1001, 7) (9706, 1, 10)
Making testing dataset...
(965, 1001, 7) (965, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
301/303 [============================>.] - ETA: 0s - loss: 0.2822 - true_positives_m: 0.5863
Epoch 1: val_true_positives_m improved from -inf to 0.54896, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_6.hdf5
303/303 [==============================] - 12s 21ms/step - loss: 0.2822 - true_positives_m: 0.5865 - val_loss: 0.2932 - val_true_positives_m: 0.5490 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
301/303 [============================>.] - ETA: 0s - loss: 0.2503 - true_positives_m: 0.6784
Epoch 2: val_true_positives_m improved from 0.54896 to 0.64688, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_6.hdf5
303/303 [==============================] - 5s 18ms/step - loss: 0.2503 - true_positives_m: 0.6781 - val_loss: 0.2619 - val_true_positives_m: 0.6469 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
301/303 [============================>.] - ETA: 0s - loss: 0.2465 - true_positives_m: 0.6948
Epoch 3: val_true_positives_m improved from 0.64688 to 0.70729, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_6.hdf5
303/303 [==============================] - 5s 17ms/step - loss: 0.2465 - true_positives_m: 0.6943 - val_loss: 0.2461 - val_true_positives_m: 0.7073 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
303/303 [==============================] - ETA: 0s - loss: 0.2457 - true_positives_m: 0.6937
Epoch 4: val_true_positives_m did not improve from 0.70729
303/303 [==============================] - 5s 17ms/step - loss: 0.2457 - true_positives_m: 0.6937 - val_loss: 0.2449 - val_true_positives_m: 0.7010 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
300/303 [============================>.] - ETA: 0s - loss: 0.2419 - true_positives_m: 0.7067
Epoch 5: val_true_positives_m improved from 0.70729 to 0.70833, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_6.hdf5
303/303 [==============================] - 5s 18ms/step - loss: 0.2420 - true_positives_m: 0.7065 - val_loss: 0.2440 - val_true_positives_m: 0.7083 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
302/303 [============================>.] - ETA: 0s - loss: 0.2397 - true_positives_m: 0.7108
Epoch 6: val_true_positives_m did not improve from 0.70833
303/303 [==============================] - 5s 17ms/step - loss: 0.2397 - true_positives_m: 0.7109 - val_loss: 0.2454 - val_true_positives_m: 0.7031 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
303/303 [==============================] - ETA: 0s - loss: 0.2385 - true_positives_m: 0.7163
Epoch 7: val_true_positives_m did not improve from 0.70833
303/303 [==============================] - 5s 17ms/step - loss: 0.2385 - true_positives_m: 0.7163 - val_loss: 0.2480 - val_true_positives_m: 0.6990 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
301/303 [============================>.] - ETA: 0s - loss: 0.2353 - true_positives_m: 0.7261
Epoch 8: val_true_positives_m did not improve from 0.70833
303/303 [==============================] - 5s 17ms/step - loss: 0.2353 - true_positives_m: 0.7259 - val_loss: 0.2452 - val_true_positives_m: 0.7052 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
303/303 [==============================] - ETA: 0s - loss: 0.2332 - true_positives_m: 0.7278
Epoch 9: val_true_positives_m improved from 0.70833 to 0.71771, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_6.hdf5
303/303 [==============================] - 5s 18ms/step - loss: 0.2332 - true_positives_m: 0.7278 - val_loss: 0.2458 - val_true_positives_m: 0.7177 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
303/303 [==============================] - ETA: 0s - loss: 0.2302 - true_positives_m: 0.7366
Epoch 10: val_true_positives_m did not improve from 0.71771
303/303 [==============================] - 5s 17ms/step - loss: 0.2302 - true_positives_m: 0.7366 - val_loss: 0.2481 - val_true_positives_m: 0.7063 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 7, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9957, 1001, 7) (9957, 1, 10)
Making testing dataset...
(1034, 1001, 7) (1034, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
309/311 [============================>.] - ETA: 0s - loss: 0.2874 - true_positives_m: 0.5537
Epoch 1: val_true_positives_m improved from -inf to 0.48633, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_7.hdf5
311/311 [==============================] - 11s 20ms/step - loss: 0.2872 - true_positives_m: 0.5545 - val_loss: 0.2897 - val_true_positives_m: 0.4863 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
309/311 [============================>.] - ETA: 0s - loss: 0.2598 - true_positives_m: 0.6374
Epoch 2: val_true_positives_m improved from 0.48633 to 0.62305, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_7.hdf5
311/311 [==============================] - 5s 18ms/step - loss: 0.2598 - true_positives_m: 0.6372 - val_loss: 0.2588 - val_true_positives_m: 0.6230 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
309/311 [============================>.] - ETA: 0s - loss: 0.2568 - true_positives_m: 0.6601
Epoch 3: val_true_positives_m improved from 0.62305 to 0.63672, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_7.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2568 - true_positives_m: 0.6595 - val_loss: 0.2619 - val_true_positives_m: 0.6367 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
310/311 [============================>.] - ETA: 0s - loss: 0.2557 - true_positives_m: 0.6646
Epoch 4: val_true_positives_m improved from 0.63672 to 0.63770, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_7.hdf5
311/311 [==============================] - 6s 18ms/step - loss: 0.2557 - true_positives_m: 0.6647 - val_loss: 0.2637 - val_true_positives_m: 0.6377 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
309/311 [============================>.] - ETA: 0s - loss: 0.2535 - true_positives_m: 0.6691
Epoch 5: val_true_positives_m improved from 0.63770 to 0.68164, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_7.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2535 - true_positives_m: 0.6688 - val_loss: 0.2573 - val_true_positives_m: 0.6816 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
309/311 [============================>.] - ETA: 0s - loss: 0.2517 - true_positives_m: 0.6734
Epoch 6: val_true_positives_m did not improve from 0.68164
311/311 [==============================] - 5s 17ms/step - loss: 0.2517 - true_positives_m: 0.6733 - val_loss: 0.2585 - val_true_positives_m: 0.6562 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
310/311 [============================>.] - ETA: 0s - loss: 0.2486 - true_positives_m: 0.6826
Epoch 7: val_true_positives_m did not improve from 0.68164
311/311 [==============================] - 5s 17ms/step - loss: 0.2486 - true_positives_m: 0.6826 - val_loss: 0.2596 - val_true_positives_m: 0.6543 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
310/311 [============================>.] - ETA: 0s - loss: 0.2479 - true_positives_m: 0.6857
Epoch 8: val_true_positives_m did not improve from 0.68164
311/311 [==============================] - 5s 17ms/step - loss: 0.2479 - true_positives_m: 0.6854 - val_loss: 0.2690 - val_true_positives_m: 0.5967 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
309/311 [============================>.] - ETA: 0s - loss: 0.2451 - true_positives_m: 0.6864
Epoch 9: val_true_positives_m did not improve from 0.68164
311/311 [==============================] - 5s 17ms/step - loss: 0.2451 - true_positives_m: 0.6871 - val_loss: 0.2575 - val_true_positives_m: 0.6553 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
311/311 [==============================] - ETA: 0s - loss: 0.2418 - true_positives_m: 0.6949
Epoch 10: val_true_positives_m did not improve from 0.68164
311/311 [==============================] - 5s 17ms/step - loss: 0.2418 - true_positives_m: 0.6949 - val_loss: 0.2587 - val_true_positives_m: 0.6670 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 7, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9957, 1001, 7) (9957, 1, 10)
Making testing dataset...
(1034, 1001, 7) (1034, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
310/311 [============================>.] - ETA: 0s - loss: 0.2865 - true_positives_m: 0.5727
Epoch 1: val_true_positives_m improved from -inf to 0.45312, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_7.hdf5
311/311 [==============================] - 11s 20ms/step - loss: 0.2865 - true_positives_m: 0.5726 - val_loss: 0.3047 - val_true_positives_m: 0.4531 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
309/311 [============================>.] - ETA: 0s - loss: 0.2597 - true_positives_m: 0.6441
Epoch 2: val_true_positives_m improved from 0.45312 to 0.64355, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_7.hdf5
311/311 [==============================] - 5s 17ms/step - loss: 0.2597 - true_positives_m: 0.6445 - val_loss: 0.2597 - val_true_positives_m: 0.6436 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
311/311 [==============================] - ETA: 0s - loss: 0.2565 - true_positives_m: 0.6533
Epoch 3: val_true_positives_m did not improve from 0.64355
311/311 [==============================] - 5s 17ms/step - loss: 0.2565 - true_positives_m: 0.6533 - val_loss: 0.2615 - val_true_positives_m: 0.6270 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
308/311 [============================>.] - ETA: 0s - loss: 0.2552 - true_positives_m: 0.6610
Epoch 4: val_true_positives_m did not improve from 0.64355
311/311 [==============================] - 5s 17ms/step - loss: 0.2552 - true_positives_m: 0.6612 - val_loss: 0.2573 - val_true_positives_m: 0.6426 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
310/311 [============================>.] - ETA: 0s - loss: 0.2521 - true_positives_m: 0.6720
Epoch 5: val_true_positives_m improved from 0.64355 to 0.64648, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_7.hdf5
311/311 [==============================] - 5s 18ms/step - loss: 0.2521 - true_positives_m: 0.6719 - val_loss: 0.2575 - val_true_positives_m: 0.6465 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
308/311 [============================>.] - ETA: 0s - loss: 0.2507 - true_positives_m: 0.6673
Epoch 6: val_true_positives_m did not improve from 0.64648
311/311 [==============================] - 5s 17ms/step - loss: 0.2507 - true_positives_m: 0.6679 - val_loss: 0.2569 - val_true_positives_m: 0.6338 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
311/311 [==============================] - ETA: 0s - loss: 0.2490 - true_positives_m: 0.6760
Epoch 7: val_true_positives_m did not improve from 0.64648
311/311 [==============================] - 5s 17ms/step - loss: 0.2490 - true_positives_m: 0.6760 - val_loss: 0.2569 - val_true_positives_m: 0.6465 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
309/311 [============================>.] - ETA: 0s - loss: 0.2471 - true_positives_m: 0.6825
Epoch 8: val_true_positives_m did not improve from 0.64648
311/311 [==============================] - 5s 17ms/step - loss: 0.2469 - true_positives_m: 0.6830 - val_loss: 0.2603 - val_true_positives_m: 0.6416 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
310/311 [============================>.] - ETA: 0s - loss: 0.2448 - true_positives_m: 0.6830
Epoch 9: val_true_positives_m did not improve from 0.64648
311/311 [==============================] - 5s 17ms/step - loss: 0.2448 - true_positives_m: 0.6828 - val_loss: 0.2596 - val_true_positives_m: 0.6387 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
309/311 [============================>.] - ETA: 0s - loss: 0.2414 - true_positives_m: 0.6888
Epoch 10: val_true_positives_m did not improve from 0.64648
311/311 [==============================] - 5s 17ms/step - loss: 0.2414 - true_positives_m: 0.6887 - val_loss: 0.2587 - val_true_positives_m: 0.6455 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 8, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9822, 1001, 7) (9822, 1, 10)
Making testing dataset...
(1029, 1001, 7) (1029, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
305/306 [============================>.] - ETA: 0s - loss: 0.2950 - true_positives_m: 0.5278
Epoch 1: val_true_positives_m improved from -inf to 0.23926, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_8.hdf5
306/306 [==============================] - 11s 21ms/step - loss: 0.2950 - true_positives_m: 0.5281 - val_loss: 0.4427 - val_true_positives_m: 0.2393 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
304/306 [============================>.] - ETA: 0s - loss: 0.2679 - true_positives_m: 0.6003
Epoch 2: val_true_positives_m improved from 0.23926 to 0.57520, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_8.hdf5
306/306 [==============================] - 6s 18ms/step - loss: 0.2679 - true_positives_m: 0.6006 - val_loss: 0.2718 - val_true_positives_m: 0.5752 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
305/306 [============================>.] - ETA: 0s - loss: 0.2649 - true_positives_m: 0.6055
Epoch 3: val_true_positives_m improved from 0.57520 to 0.60938, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_8.hdf5
306/306 [==============================] - 5s 18ms/step - loss: 0.2649 - true_positives_m: 0.6056 - val_loss: 0.2697 - val_true_positives_m: 0.6094 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
306/306 [==============================] - ETA: 0s - loss: 0.2634 - true_positives_m: 0.6197
Epoch 4: val_true_positives_m did not improve from 0.60938
306/306 [==============================] - 5s 17ms/step - loss: 0.2634 - true_positives_m: 0.6197 - val_loss: 0.2727 - val_true_positives_m: 0.5986 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
304/306 [============================>.] - ETA: 0s - loss: 0.2614 - true_positives_m: 0.6167
Epoch 5: val_true_positives_m improved from 0.60938 to 0.62109, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_8.hdf5
306/306 [==============================] - 5s 18ms/step - loss: 0.2613 - true_positives_m: 0.6172 - val_loss: 0.2697 - val_true_positives_m: 0.6211 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
303/306 [============================>.] - ETA: 0s - loss: 0.2594 - true_positives_m: 0.6267
Epoch 6: val_true_positives_m improved from 0.62109 to 0.62305, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_8.hdf5
306/306 [==============================] - 5s 18ms/step - loss: 0.2595 - true_positives_m: 0.6259 - val_loss: 0.2735 - val_true_positives_m: 0.6230 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
305/306 [============================>.] - ETA: 0s - loss: 0.2574 - true_positives_m: 0.6328
Epoch 7: val_true_positives_m did not improve from 0.62305
306/306 [==============================] - 5s 17ms/step - loss: 0.2573 - true_positives_m: 0.6329 - val_loss: 0.2710 - val_true_positives_m: 0.6221 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
304/306 [============================>.] - ETA: 0s - loss: 0.2551 - true_positives_m: 0.6380
Epoch 8: val_true_positives_m did not improve from 0.62305
306/306 [==============================] - 5s 17ms/step - loss: 0.2550 - true_positives_m: 0.6376 - val_loss: 0.2711 - val_true_positives_m: 0.6211 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
303/306 [============================>.] - ETA: 0s - loss: 0.2524 - true_positives_m: 0.6471
Epoch 9: val_true_positives_m did not improve from 0.62305
306/306 [==============================] - 5s 17ms/step - loss: 0.2524 - true_positives_m: 0.6470 - val_loss: 0.2716 - val_true_positives_m: 0.6113 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
304/306 [============================>.] - ETA: 0s - loss: 0.2495 - true_positives_m: 0.6520
Epoch 10: val_true_positives_m did not improve from 0.62305
306/306 [==============================] - 5s 17ms/step - loss: 0.2495 - true_positives_m: 0.6522 - val_loss: 0.2742 - val_true_positives_m: 0.6055 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 8, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(9822, 1001, 7) (9822, 1, 10)
Making testing dataset...
(1029, 1001, 7) (1029, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
304/306 [============================>.] - ETA: 0s - loss: 0.2960 - true_positives_m: 0.5199
Epoch 1: val_true_positives_m improved from -inf to 0.54785, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_8.hdf5
306/306 [==============================] - 11s 21ms/step - loss: 0.2959 - true_positives_m: 0.5203 - val_loss: 0.2796 - val_true_positives_m: 0.5479 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
306/306 [==============================] - ETA: 0s - loss: 0.2669 - true_positives_m: 0.6009
Epoch 2: val_true_positives_m improved from 0.54785 to 0.56934, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_8.hdf5
306/306 [==============================] - 6s 19ms/step - loss: 0.2669 - true_positives_m: 0.6009 - val_loss: 0.2707 - val_true_positives_m: 0.5693 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
305/306 [============================>.] - ETA: 0s - loss: 0.2644 - true_positives_m: 0.6120
Epoch 3: val_true_positives_m improved from 0.56934 to 0.58496, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_8.hdf5
306/306 [==============================] - 6s 18ms/step - loss: 0.2645 - true_positives_m: 0.6120 - val_loss: 0.2684 - val_true_positives_m: 0.5850 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
304/306 [============================>.] - ETA: 0s - loss: 0.2615 - true_positives_m: 0.6199
Epoch 4: val_true_positives_m did not improve from 0.58496
306/306 [==============================] - 5s 18ms/step - loss: 0.2615 - true_positives_m: 0.6203 - val_loss: 0.2699 - val_true_positives_m: 0.5840 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
304/306 [============================>.] - ETA: 0s - loss: 0.2598 - true_positives_m: 0.6176
Epoch 5: val_true_positives_m improved from 0.58496 to 0.61230, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_8.hdf5
306/306 [==============================] - 6s 18ms/step - loss: 0.2597 - true_positives_m: 0.6182 - val_loss: 0.2679 - val_true_positives_m: 0.6123 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
304/306 [============================>.] - ETA: 0s - loss: 0.2567 - true_positives_m: 0.6270
Epoch 6: val_true_positives_m did not improve from 0.61230
306/306 [==============================] - 5s 18ms/step - loss: 0.2567 - true_positives_m: 0.6270 - val_loss: 0.2692 - val_true_positives_m: 0.6055 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
306/306 [==============================] - ETA: 0s - loss: 0.2536 - true_positives_m: 0.6452
Epoch 7: val_true_positives_m did not improve from 0.61230
306/306 [==============================] - 5s 18ms/step - loss: 0.2536 - true_positives_m: 0.6452 - val_loss: 0.2710 - val_true_positives_m: 0.6055 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
306/306 [==============================] - ETA: 0s - loss: 0.2504 - true_positives_m: 0.6463
Epoch 8: val_true_positives_m did not improve from 0.61230
306/306 [==============================] - 5s 18ms/step - loss: 0.2504 - true_positives_m: 0.6463 - val_loss: 0.2722 - val_true_positives_m: 0.6094 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
304/306 [============================>.] - ETA: 0s - loss: 0.2464 - true_positives_m: 0.6602
Epoch 9: val_true_positives_m did not improve from 0.61230
306/306 [==============================] - 6s 18ms/step - loss: 0.2464 - true_positives_m: 0.6594 - val_loss: 0.2741 - val_true_positives_m: 0.5947 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
303/306 [============================>.] - ETA: 0s - loss: 0.2437 - true_positives_m: 0.6555
Epoch 10: val_true_positives_m did not improve from 0.61230
306/306 [==============================] - 5s 18ms/step - loss: 0.2436 - true_positives_m: 0.6549 - val_loss: 0.2742 - val_true_positives_m: 0.5977 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 9, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10053, 1001, 7) (10053, 1, 10)
Making testing dataset...
(990, 1001, 7) (990, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
311/314 [============================>.] - ETA: 0s - loss: 0.3017 - true_positives_m: 0.4962
Epoch 1: val_true_positives_m improved from -inf to 0.32188, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_9.hdf5
314/314 [==============================] - 11s 20ms/step - loss: 0.3015 - true_positives_m: 0.4974 - val_loss: 0.3436 - val_true_positives_m: 0.3219 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
312/314 [============================>.] - ETA: 0s - loss: 0.2782 - true_positives_m: 0.5548
Epoch 2: val_true_positives_m improved from 0.32188 to 0.52917, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_9.hdf5
314/314 [==============================] - 6s 18ms/step - loss: 0.2783 - true_positives_m: 0.5546 - val_loss: 0.2835 - val_true_positives_m: 0.5292 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
314/314 [==============================] - ETA: 0s - loss: 0.2768 - true_positives_m: 0.5569
Epoch 3: val_true_positives_m improved from 0.52917 to 0.55625, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_9.hdf5
314/314 [==============================] - 6s 18ms/step - loss: 0.2768 - true_positives_m: 0.5569 - val_loss: 0.2790 - val_true_positives_m: 0.5562 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
312/314 [============================>.] - ETA: 0s - loss: 0.2744 - true_positives_m: 0.5662
Epoch 4: val_true_positives_m improved from 0.55625 to 0.56771, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_9.hdf5
314/314 [==============================] - 5s 17ms/step - loss: 0.2746 - true_positives_m: 0.5659 - val_loss: 0.2789 - val_true_positives_m: 0.5677 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
311/314 [============================>.] - ETA: 0s - loss: 0.2732 - true_positives_m: 0.5764
Epoch 5: val_true_positives_m did not improve from 0.56771
314/314 [==============================] - 5s 17ms/step - loss: 0.2732 - true_positives_m: 0.5766 - val_loss: 0.2797 - val_true_positives_m: 0.5615 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
312/314 [============================>.] - ETA: 0s - loss: 0.2720 - true_positives_m: 0.5832
Epoch 6: val_true_positives_m did not improve from 0.56771
314/314 [==============================] - 5s 17ms/step - loss: 0.2721 - true_positives_m: 0.5833 - val_loss: 0.2810 - val_true_positives_m: 0.5375 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
314/314 [==============================] - ETA: 0s - loss: 0.2706 - true_positives_m: 0.5809
Epoch 7: val_true_positives_m did not improve from 0.56771
314/314 [==============================] - 5s 17ms/step - loss: 0.2706 - true_positives_m: 0.5809 - val_loss: 0.2827 - val_true_positives_m: 0.5531 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
314/314 [==============================] - ETA: 0s - loss: 0.2686 - true_positives_m: 0.5878
Epoch 8: val_true_positives_m did not improve from 0.56771
314/314 [==============================] - 5s 17ms/step - loss: 0.2686 - true_positives_m: 0.5878 - val_loss: 0.2846 - val_true_positives_m: 0.5344 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
312/314 [============================>.] - ETA: 0s - loss: 0.2654 - true_positives_m: 0.5956
Epoch 9: val_true_positives_m improved from 0.56771 to 0.56875, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_9.hdf5
314/314 [==============================] - 5s 17ms/step - loss: 0.2655 - true_positives_m: 0.5956 - val_loss: 0.2809 - val_true_positives_m: 0.5688 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
314/314 [==============================] - ETA: 0s - loss: 0.2638 - true_positives_m: 0.5985
Epoch 10: val_true_positives_m did not improve from 0.56875
314/314 [==============================] - 5s 17ms/step - loss: 0.2638 - true_positives_m: 0.5985 - val_loss: 0.2849 - val_true_positives_m: 0.5521 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 9, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10053, 1001, 7) (10053, 1, 10)
Making testing dataset...
(990, 1001, 7) (990, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
311/314 [============================>.] - ETA: 0s - loss: 0.3018 - true_positives_m: 0.4996
Epoch 1: val_true_positives_m improved from -inf to 0.41771, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_9.hdf5
314/314 [==============================] - 11s 20ms/step - loss: 0.3017 - true_positives_m: 0.4996 - val_loss: 0.3231 - val_true_positives_m: 0.4177 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
312/314 [============================>.] - ETA: 0s - loss: 0.2783 - true_positives_m: 0.5473
Epoch 2: val_true_positives_m improved from 0.41771 to 0.55625, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_9.hdf5
314/314 [==============================] - 6s 18ms/step - loss: 0.2783 - true_positives_m: 0.5473 - val_loss: 0.2815 - val_true_positives_m: 0.5562 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
312/314 [============================>.] - ETA: 0s - loss: 0.2753 - true_positives_m: 0.5587
Epoch 3: val_true_positives_m did not improve from 0.55625
314/314 [==============================] - 5s 17ms/step - loss: 0.2752 - true_positives_m: 0.5587 - val_loss: 0.2821 - val_true_positives_m: 0.5156 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
313/314 [============================>.] - ETA: 0s - loss: 0.2733 - true_positives_m: 0.5640
Epoch 4: val_true_positives_m did not improve from 0.55625
314/314 [==============================] - 5s 17ms/step - loss: 0.2732 - true_positives_m: 0.5644 - val_loss: 0.2802 - val_true_positives_m: 0.5531 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
313/314 [============================>.] - ETA: 0s - loss: 0.2719 - true_positives_m: 0.5691
Epoch 5: val_true_positives_m did not improve from 0.55625
314/314 [==============================] - 5s 17ms/step - loss: 0.2718 - true_positives_m: 0.5690 - val_loss: 0.2793 - val_true_positives_m: 0.5469 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
311/314 [============================>.] - ETA: 0s - loss: 0.2697 - true_positives_m: 0.5841
Epoch 6: val_true_positives_m did not improve from 0.55625
314/314 [==============================] - 5s 17ms/step - loss: 0.2696 - true_positives_m: 0.5836 - val_loss: 0.2807 - val_true_positives_m: 0.5469 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
312/314 [============================>.] - ETA: 0s - loss: 0.2675 - true_positives_m: 0.5843
Epoch 7: val_true_positives_m improved from 0.55625 to 0.56458, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_9.hdf5
314/314 [==============================] - 5s 17ms/step - loss: 0.2673 - true_positives_m: 0.5844 - val_loss: 0.2807 - val_true_positives_m: 0.5646 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
313/314 [============================>.] - ETA: 0s - loss: 0.2644 - true_positives_m: 0.5917
Epoch 8: val_true_positives_m did not improve from 0.56458
314/314 [==============================] - 5s 17ms/step - loss: 0.2644 - true_positives_m: 0.5917 - val_loss: 0.2817 - val_true_positives_m: 0.5552 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
313/314 [============================>.] - ETA: 0s - loss: 0.2621 - true_positives_m: 0.6023
Epoch 9: val_true_positives_m did not improve from 0.56458
314/314 [==============================] - 5s 17ms/step - loss: 0.2620 - true_positives_m: 0.6026 - val_loss: 0.2822 - val_true_positives_m: 0.5542 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
313/314 [============================>.] - ETA: 0s - loss: 0.2589 - true_positives_m: 0.6065
Epoch 10: val_true_positives_m did not improve from 0.56458
314/314 [==============================] - 5s 17ms/step - loss: 0.2590 - true_positives_m: 0.6061 - val_loss: 0.2833 - val_true_positives_m: 0.5323 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 10, Net Number: 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10093, 1001, 7) (10093, 1, 10)
Making testing dataset...
(1020, 1001, 7) (1020, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
313/315 [============================>.] - ETA: 0s - loss: 0.3174 - true_positives_m: 0.4475
Epoch 1: val_true_positives_m improved from -inf to 0.38508, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 11s 21ms/step - loss: 0.3172 - true_positives_m: 0.4480 - val_loss: 0.3248 - val_true_positives_m: 0.3851 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
313/315 [============================>.] - ETA: 0s - loss: 0.2881 - true_positives_m: 0.5067
Epoch 2: val_true_positives_m improved from 0.38508 to 0.47883, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 19ms/step - loss: 0.2881 - true_positives_m: 0.5066 - val_loss: 0.2957 - val_true_positives_m: 0.4788 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
312/315 [============================>.] - ETA: 0s - loss: 0.2849 - true_positives_m: 0.5171
Epoch 3: val_true_positives_m improved from 0.47883 to 0.50605, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2849 - true_positives_m: 0.5176 - val_loss: 0.2909 - val_true_positives_m: 0.5060 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
313/315 [============================>.] - ETA: 0s - loss: 0.2826 - true_positives_m: 0.5292
Epoch 4: val_true_positives_m improved from 0.50605 to 0.50706, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2827 - true_positives_m: 0.5287 - val_loss: 0.2922 - val_true_positives_m: 0.5071 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
315/315 [==============================] - ETA: 0s - loss: 0.2799 - true_positives_m: 0.5411
Epoch 5: val_true_positives_m improved from 0.50706 to 0.51210, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2799 - true_positives_m: 0.5411 - val_loss: 0.2916 - val_true_positives_m: 0.5121 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
315/315 [==============================] - ETA: 0s - loss: 0.2782 - true_positives_m: 0.5464
Epoch 6: val_true_positives_m did not improve from 0.51210
315/315 [==============================] - 6s 18ms/step - loss: 0.2782 - true_positives_m: 0.5464 - val_loss: 0.2923 - val_true_positives_m: 0.5111 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
315/315 [==============================] - ETA: 0s - loss: 0.2758 - true_positives_m: 0.5569
Epoch 7: val_true_positives_m improved from 0.51210 to 0.52419, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2758 - true_positives_m: 0.5569 - val_loss: 0.2918 - val_true_positives_m: 0.5242 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
315/315 [==============================] - ETA: 0s - loss: 0.2730 - true_positives_m: 0.5631
Epoch 8: val_true_positives_m did not improve from 0.52419
315/315 [==============================] - 5s 17ms/step - loss: 0.2730 - true_positives_m: 0.5631 - val_loss: 0.2928 - val_true_positives_m: 0.5040 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
312/315 [============================>.] - ETA: 0s - loss: 0.2713 - true_positives_m: 0.5699
Epoch 9: val_true_positives_m did not improve from 0.52419
315/315 [==============================] - 6s 18ms/step - loss: 0.2712 - true_positives_m: 0.5690 - val_loss: 0.2935 - val_true_positives_m: 0.5171 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
313/315 [============================>.] - ETA: 0s - loss: 0.2679 - true_positives_m: 0.5797
Epoch 10: val_true_positives_m improved from 0.52419 to 0.52520, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_1_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2678 - true_positives_m: 0.5799 - val_loss: 0.2943 - val_true_positives_m: 0.5252 - lr: 1.0000e-04
Learning finished
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class: 10, Net Number: 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FRC_CLASSES [0.001 0.002 0.003 0.006 0.01  0.015 0.025 0.039 0.063 0.1  ]
First learning rate:  0.001
Model: "kint"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 1001, 16)          1024      
                                                                 
 conv1d_1 (Conv1D)           (None, 1001, 16)          2320      
                                                                 
 conv1d_2 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_3 (Conv1D)           (None, 500, 16)           2320      
                                                                 
 conv1d_4 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_5 (Conv1D)           (None, 250, 16)           2320      
                                                                 
 conv1d_6 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 conv1d_7 (Conv1D)           (None, 50, 16)            2320      
                                                                 
 batch_normalization (BatchN  (None, 1001, 16)         64        
 ormalization)                                                   
                                                                 
 batch_normalization_1 (Batc  (None, 1001, 16)         64        
 hNormalization)                                                 
                                                                 
 batch_normalization_2 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_3 (Batc  (None, 500, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_4 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_5 (Batc  (None, 250, 16)          64        
 hNormalization)                                                 
                                                                 
 batch_normalization_6 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 batch_normalization_7 (Batc  (None, 50, 16)           64        
 hNormalization)                                                 
                                                                 
 max_pooling1d (MaxPooling1D  (None, 500, 16)          0         
 )                                                               
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 250, 16)          0         
 1D)                                                             
                                                                 
 max_pooling1d_2 (MaxPooling  (None, 50, 16)           0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 800)               0         
                                                                 
 dense (Dense)               (None, 256)               205056    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 257,018
Trainable params: 256,762
Non-trainable params: 256
_________________________________________________________________
Making training dataset...
(10093, 1001, 7) (10093, 1, 10)
Making testing dataset...
(1020, 1001, 7) (1020, 1, 10)
Training...
LEARNING RATE: 0.001
Epoch 1/10
312/315 [============================>.] - ETA: 0s - loss: 0.3069 - true_positives_m: 0.4665
Epoch 1: val_true_positives_m improved from -inf to 0.28226, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_10.hdf5
315/315 [==============================] - 12s 21ms/step - loss: 0.3069 - true_positives_m: 0.4660 - val_loss: 0.3667 - val_true_positives_m: 0.2823 - lr: 0.0010
LEARNING RATE: 0.001
Epoch 2/10
314/315 [============================>.] - ETA: 0s - loss: 0.2869 - true_positives_m: 0.5189
Epoch 2: val_true_positives_m improved from 0.28226 to 0.50302, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2869 - true_positives_m: 0.5190 - val_loss: 0.2918 - val_true_positives_m: 0.5030 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 3/10
312/315 [============================>.] - ETA: 0s - loss: 0.2851 - true_positives_m: 0.5217
Epoch 3: val_true_positives_m improved from 0.50302 to 0.52621, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2850 - true_positives_m: 0.5229 - val_loss: 0.2876 - val_true_positives_m: 0.5262 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 4/10
315/315 [==============================] - ETA: 0s - loss: 0.2834 - true_positives_m: 0.5332
Epoch 4: val_true_positives_m did not improve from 0.52621
315/315 [==============================] - 5s 17ms/step - loss: 0.2834 - true_positives_m: 0.5332 - val_loss: 0.2881 - val_true_positives_m: 0.5141 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 5/10
314/315 [============================>.] - ETA: 0s - loss: 0.2826 - true_positives_m: 0.5374
Epoch 5: val_true_positives_m did not improve from 0.52621
315/315 [==============================] - 5s 17ms/step - loss: 0.2827 - true_positives_m: 0.5374 - val_loss: 0.2893 - val_true_positives_m: 0.5252 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 6/10
313/315 [============================>.] - ETA: 0s - loss: 0.2806 - true_positives_m: 0.5418
Epoch 6: val_true_positives_m improved from 0.52621 to 0.53931, saving model to /home/avshmelev/bash_scripts/rnn\weights_net_num_2_frc_class_10.hdf5
315/315 [==============================] - 6s 18ms/step - loss: 0.2807 - true_positives_m: 0.5417 - val_loss: 0.2878 - val_true_positives_m: 0.5393 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 7/10
314/315 [============================>.] - ETA: 0s - loss: 0.2782 - true_positives_m: 0.5463
Epoch 7: val_true_positives_m did not improve from 0.53931
315/315 [==============================] - 5s 17ms/step - loss: 0.2782 - true_positives_m: 0.5462 - val_loss: 0.2948 - val_true_positives_m: 0.5030 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 8/10
315/315 [==============================] - ETA: 0s - loss: 0.2761 - true_positives_m: 0.5590
Epoch 8: val_true_positives_m did not improve from 0.53931
315/315 [==============================] - 5s 17ms/step - loss: 0.2761 - true_positives_m: 0.5590 - val_loss: 0.2894 - val_true_positives_m: 0.5222 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 9/10
312/315 [============================>.] - ETA: 0s - loss: 0.2734 - true_positives_m: 0.5572
Epoch 9: val_true_positives_m did not improve from 0.53931
315/315 [==============================] - 5s 17ms/step - loss: 0.2734 - true_positives_m: 0.5575 - val_loss: 0.2899 - val_true_positives_m: 0.5353 - lr: 1.0000e-04
LEARNING RATE: 0.0001
Epoch 10/10
315/315 [==============================] - ETA: 0s - loss: 0.2699 - true_positives_m: 0.5762
Epoch 10: val_true_positives_m did not improve from 0.53931
315/315 [==============================] - 6s 18ms/step - loss: 0.2699 - true_positives_m: 0.5762 - val_loss: 0.2920 - val_true_positives_m: 0.5101 - lr: 1.0000e-04
Learning finished

Process finished with exit code 0

```
