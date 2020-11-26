import tensorflow as tf

class simpleClassifierModel(tf.keras.Model):

    def __init__(self, in_shape, base_hidden_units, weight_decay):
        super(simpleClassifierModel, self).__init__()
        self.in_shape = in_shape
        self.base_hidden_units = base_hidden_units
        self.weight_decay = weight_decay
        self.inputs = tf.keras.Input(shape=self.in_shape)
        self.conv1 = tf.keras.layers.Conv2D( filters=self.base_hidden_units,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.conv2 = tf.keras.layers.Conv2D( filters=self.base_hidden_units,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.conv3 = tf.keras.layers.Conv2D( filters=self.base_hidden_units*2,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.conv4 = tf.keras.layers.Conv2D( filters=self.base_hidden_units*2,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.conv5 = tf.keras.layers.Conv2D( filters=self.base_hidden_units*4,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.conv6 = tf.keras.layers.Conv2D( filters=self.base_hidden_units*4,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            activation='relu',
            input_shape=in_shape[1:] )
        self.fc7 = tf.keras.layers.Flatten()
        self.outputs = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(.2)(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(.3)(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(.4)(x)
        x = self.fc7(x)
        outputs = self.outputs(x)
        
        #if training:
        #  x = self.dropout(x, training=training)
        return outputs


