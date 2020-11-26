from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau


# prepare the data from traininig
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]
print('x_train =', x_train.shape)
print('x_valid =', x_valid.shape)
print('x_test =', x_test.shape)

# normalize the data
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_valid = (x_valid-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

# one-hot encode the labels
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid,num_classes)
y_test =  tf.keras.utils.to_categorical(y_test,num_classes)

# data augmentation
datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
            )
datagen.fit(x_train)


''' Define the network architecture '''
base_hidden_units = 32
weight_decay = 1e-4

# INPUT LAYER
input_shape = (32, 32, 3)
inputs = tf.keras.Input(shape=input_shape)

# CONV1
x = tf.keras.layers.Conv2D( filters=base_hidden_units,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(inputs)
x = tf.keras.layers.BatchNormalization()(x)

# CONV2
x = tf.keras.layers.Conv2D( filters=base_hidden_units,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(x)
x = tf.keras.layers.BatchNormalization()(x)

# POOL + DROPOUT
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Dropout(.2)(x)

# CONV3
x = tf.keras.layers.Conv2D( filters=base_hidden_units*2,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(x)
x = tf.keras.layers.BatchNormalization()(x)

# CONV4
x = tf.keras.layers.Conv2D( filters=base_hidden_units*2,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(x)
x = tf.keras.layers.BatchNormalization()(x)

# POOL + DROPOUT
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Dropout(.3)(x)

# CONV5
x = tf.keras.layers.Conv2D( filters=base_hidden_units*4,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(x)
x = tf.keras.layers.BatchNormalization()(x)

# CONV6
x = tf.keras.layers.Conv2D( filters=base_hidden_units*4,
    kernel_size=3,
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    activation='relu',
    input_shape=input_shape[1:] )(x)
x = tf.keras.layers.BatchNormalization()(x)

# POOL + DROPOUT
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Dropout(.4)(x)

# FC7
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

''' Train the model '''
batch_size = 128
epochs = 125

checkpointer = ModelCheckpoint(filepath='model.100epochs.hdf5', verbose=1, save_best_only=True )
early_stopping = EarlyStopping( monitor='val_loss',
                                patience=10,
                                verbose=1 )
reduce_lr = ReduceLROnPlateau( monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1 )
callbacks = [ checkpointer,
              tf.keras.callbacks.ModelCheckpoint("./saves/save_at_{epoch}.h5", save_freq=5),
              TensorBoard(log_dir='logs/{}'.format(time())),
              early_stopping,
              reduce_lr, ]
optimizer = tf.keras.optimizers.Adam(lr=0.0001,decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(datagen.flow( x_train,
                                  y_train, batch_size=batch_size),
                                  callbacks=callbacks,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(x_valid, y_valid) )

# save the model, weights, etc
model.save('trained_model.h5')
