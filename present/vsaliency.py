import keras

# reznet = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(51, 51, 1), pooling=None, classes=1000)
# out = reznet.get_layer('conv5_block3_add').output
# out = keras.layers.Flatten()(out)
# out = keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001))(out)
# out = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001))(out)
# out = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001))(out)
# out = keras.layers.Dense(9)(out)
# model = keras.Model(reznet.input, out)
#
# import numpy as np
# a = np.zeros((51,51,1))
# model.predict(a.reshape(1,51,51,1))
# print(model.summary())
from  keras import layers, regularizers
in1 = layers.Input((51, 51, 1,))
m1 = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu', input_shape=(51, 51, 1))(in1)
m1 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(m1)
m1 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(m1)
m1 = layers.Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(m1)
m1 = layers.Flatten()(m1)
conv_model = keras.Model(in1, m1)

out = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(m1)
out = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(out)
out = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(out)
out = layers.Dense(9)(out)

model = keras.Model(in1, out)
model.summary()