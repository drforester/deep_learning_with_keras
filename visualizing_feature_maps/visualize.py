'''
Here we vizualize the feature maps of the layers of the "cat_dog_model" that we trained in "from_data_prep_to_model_evaluation".
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


trained_model = keras.models.load_model('cat_dog_model.h5')
layer_names = [layer.name for layer in trained_model.layers]
#print(layer_names)
#['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', #'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', #'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', #'block5_pool', 'flatten', 'FC_2', 'batch_normalization', 'dropout', 'softmax']

# what are the outputs of the model layers?
layer_outputs = [layer.output for layer in trained_model.layers]
#for x in layer_outputs:
#    print(x)
#Tensor("input_1:0", shape=(None, 224, 224, 3), dtype=float32)
#Tensor("block1_conv1/Relu:0", shape=(None, 224, 224, 64), dtype=float32)
#Tensor("block1_conv2/Relu:0", shape=(None, 224, 224, 64), dtype=float32)
#Tensor("block1_pool/MaxPool:0", shape=(None, 112, 112, 64), dtype=float32)
#Tensor("block2_conv1/Relu:0", shape=(None, 112, 112, 128), dtype=float32)
#Tensor("block2_conv2/Relu:0", shape=(None, 112, 112, 128), dtype=float32)
#Tensor("block2_pool/MaxPool:0", shape=(None, 56, 56, 128), dtype=float32)
#Tensor("block3_conv1/Relu:0", shape=(None, 56, 56, 256), dtype=float32)
#Tensor("block3_conv2/Relu:0", shape=(None, 56, 56, 256), dtype=float32)
#Tensor("block3_conv3/Relu:0", shape=(None, 56, 56, 256), dtype=float32)
#Tensor("block3_pool/MaxPool:0", shape=(None, 28, 28, 256), dtype=float32)
#Tensor("block4_conv1/Relu:0", shape=(None, 28, 28, 512), dtype=float32)
#Tensor("block4_conv2/Relu:0", shape=(None, 28, 28, 512), dtype=float32)
#Tensor("block4_conv3/Relu:0", shape=(None, 28, 28, 512), dtype=float32)
#Tensor("block4_pool/MaxPool:0", shape=(None, 14, 14, 512), dtype=float32)
#Tensor("block5_conv1/Relu:0", shape=(None, 14, 14, 512), dtype=float32)
#Tensor("block5_conv2/Relu:0", shape=(None, 14, 14, 512), dtype=float32)
#Tensor("block5_conv3/Relu:0", shape=(None, 14, 14, 512), dtype=float32)
#Tensor("block5_pool/MaxPool:0", shape=(None, 7, 7, 512), dtype=float32)
#Tensor("flatten/Reshape:0", shape=(None, 25088), dtype=float32)
#Tensor("FC_2/Relu:0", shape=(None, 64), dtype=float32)
#Tensor("batch_normalization/batchnorm/add_1:0", shape=(None, 64), dtype=float32)
#Tensor("dropout/cond/Identity:0", shape=(None, 64), dtype=float32)
#Tensor("softmax/Softmax:0", shape=(None, 2), dtype=float32)

# select only the convolutional layers
layer_outputs = []               
layer_names = []
for layer in trained_model.layers:
    if '_conv' in layer.name:
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)                   

''' Now, prepare an image to give it as an input to the above feature_map_model '''
img_path = './cat.12260.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_tensor = keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

activation_model = keras.models.Model(inputs=trained_model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    print('layer_name:', layer_name)
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            
            # post-process this image to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
        


