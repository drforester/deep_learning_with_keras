'''
to use tensorboard, run 
    ~$ tensorboard --logdir=logs/
from the same dir as this training file.
Then connect yiour browser to the localhost URL returned.
'''


from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard

train_path  = 'data/train'
valid_path  = 'data/valid'

batch_size = 32

train_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(
    train_path, target_size=(224,224), batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(
    valid_path, target_size=(224,224), batch_size=batch_size)

base_model = vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (224,224, 3))

# iterate through its layers and lock them to make them not trainable with this code
for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras import layers, models

# use “get_layer” method to save the last layer of the network
# save the output of the last layer to be the input of the next layer
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

# flatten the classifier input which is output of the last layer of VGG16 model
x = layers.Flatten()(last_output)

# add 2 FC layers, each has 4096 units and relu activation 
x = layers.Dense(64, activation='relu', name='FC_2')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(2, activation='softmax', name='softmax')(x)

# instantiate a new_model using keras’s Model class
new_model = models.Model(inputs=base_model.input, outputs=x)

# load a weights from previous training epochs?
#new_model.load_weights('./saves/save_at_2.h5')    

new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 5
train_steps = train_batches.n//train_batches.batch_size
val_steps = valid_batches.n//valid_batches.batch_size

callbacks = [ keras.callbacks.ModelCheckpoint("./saves/save_at_{epoch}.h5", period=5),
               TensorBoard(log_dir='logs/{}'.format(time())) ]
new_model.fit( train_batches, steps_per_epoch=train_steps, validation_data=valid_batches,
               validation_steps=val_steps, epochs=epochs, verbose=1, callbacks=callbacks )

# save the model, weights, etc
new_model.save('cat_dog_model.h5')


