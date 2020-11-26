import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from tensorflow.keras import metrics
from sklearn.metrics import classification_report, confusion_matrix

test_path  = 'data/test'

batch_size = 32
test_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input).flow_from_directory(
    test_path, target_size=(224,224), batch_size=batch_size, shuffle=False)

int_labels = test_batches.labels

trained_model = keras.models.load_model('cat_dog_model.h5')

y_pred = trained_model.predict(test_batches, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(" --- confusion matrix --- ")
print(confusion_matrix(int_labels, y_pred_bool))
print('')

## one could do it like this, using integer labels
#print(classification_report(int_labels, y_pred_bool))

# or like this using the class names
name_list = list(test_batches.class_indices.keys())
name_labels = np.array([name_list[x] for x in int_labels])
y_pred_names = np.array([name_list[x] for x in y_pred_bool])
print(classification_report(name_labels, y_pred_names))

