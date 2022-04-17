# Developing derictory
Link for training and testing data downloading: [Google Drive](https://drive.google.com/drive/folders/1x6GPoihKo1m2J6SnnCaCmT0-b3f2x0V9?usp=sharing)

# Start cluster shell

```
srun --pty --mincpus=2 --gpus=1 bash
```

# Compile SELAM

```
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
