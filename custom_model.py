import numpy as np
from six.moves import range
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adadelta
from keras import callbacks
import h5py
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

weights_file = "saved/custom_weights_48.h5"


def subtract_mean(img):
    for i in range(img.shape[0]):
        img[i] -= img[i].mean()
    return img


h5f = h5py.File('data/SVHN_GREY_48.h5','r')


X_train = h5f['train_dataset'][:]
y_train = h5f['train_labels'][:]
X_val = h5f['valid_dataset'][:]
y_val = h5f['valid_labels'][:]
X_test = h5f['test_dataset'][:]
y_test = h5f['test_labels'][:]


h5f.close()



h5f2 = h5py.File('data/SVHN_neg_GRAY_int16_48.h5','r')


X_train_neg = h5f2['train_dataset_neg'][:]
y_train_neg = h5f2['train_labels_neg'][:]


h5f2.close()

#Add Negative Examples to the classifier
X_train = np.vstack((X_train, X_train_neg))
y_train = np.vstack((y_train, y_train_neg))

#Shuffle Examples
p = np.random.permutation(y_train.shape[0])
X_train = X_train[p]
y_train = y_train[p]



X_train = subtract_mean(X_train.astype('float32'))
print("X Val: ", X_val[0,0])
X_val2 = subtract_mean(X_val.astype('float32'))

print("X TRAIN", X_train[1:10])
print("y TRAIN", y_train[1:10])
print("X TRAIN SHAPE", X_train.shape)
print("y TRAIN SHAPE", y_train.shape)

y_temp = np.copy(y_train).transpose()
y0 = y_temp[0]
y1 = y_temp[1]
y2 = y_temp[2]
y3 = y_temp[3]
y4 = y_temp[4]

y_train2 = [y0, y1, y2, y3, y4]

y_temp = np.copy(y_val).transpose()
y0 = y_temp[0]
y1 = y_temp[1]
y2 = y_temp[2]
y3 = y_temp[3]
y4 = y_temp[4]

y_val2 = [y0, y1, y2, y3, y4]

depth1 = 48
depth2 = 64
depth3 = 80
depth4 = 96
depth5 = 128
depth6 = 128

kernel_size = (3,3)
kernel_init = 'random_uniform'
kernel_constraint = maxnorm(2.)
pad_type = 'same'
num_classes = 11

input_shape = Input(shape = (48, 48, 1))

x = Convolution2D(filters=depth1, kernel_size=kernel_size, kernel_initializer=kernel_init, \
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b1_conv1")(input_shape)
x = Activation('relu', name="b1_act1")(x)
x = BatchNormalization(name="b1_bnorm1")(x)
x = MaxPooling2D(name="b1_pool1")(x)
x = Dropout(0.25, name="b1_drop1")(x)

x = Convolution2D(filters=depth2, kernel_size=kernel_size, kernel_initializer=kernel_init, \
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b2_conv1")(x)
x = Activation('relu', name="b2_act1")(x)
x = BatchNormalization(name="b2_bnorm1")(x)
x = Dropout(0.25, name="b2_drop1")(x)


x = Convolution2D(filters=depth3, kernel_size=kernel_size, kernel_initializer=kernel_init,\
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b2_conv2")(x)
x = Activation('relu', name="b2_act2")(x)
x = BatchNormalization(name="b2_bnorm2")(x)
x = MaxPooling2D(name="b2_pool1")(x)
x = Dropout(0.25, name="b2_drop2")(x)

x = Convolution2D(filters=depth4, kernel_size=kernel_size, kernel_initializer=kernel_init, \
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b3_conv1")(x)
x = Activation('relu', name="b3_act1")(x)
x = BatchNormalization(name="b3_bnorm1")(x)
x = MaxPooling2D(name="b3_pool1")(x)
x = Dropout(0.25, name="b3_drop1")(x)


x = Convolution2D(filters=depth5, kernel_size=kernel_size, kernel_initializer=kernel_init, \
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b3_conv2")(x)
x = Activation('relu', name="b3_act2")(x)
x = BatchNormalization(name="b3_bnorm2")(x)
x = MaxPooling2D(name="b3_pool2")(x)
x = Dropout(0.25, name="b3_drop2")(x)

x = Convolution2D(filters=depth6, kernel_size=kernel_size, kernel_initializer=kernel_init, \
                  kernel_constraint=kernel_constraint, padding=pad_type, name="b4_conv1")(x)
x = Activation('relu', name="b4_act1")(x)
x = BatchNormalization(name="b4_bnorm1")(x)
x = Dropout(0.5, name="b4_drop1")(x)


flat_layer = Flatten(name="feature_vector")(x)

digit1 = Dense(units = num_classes, activation="softmax", name="output1")(flat_layer)
digit2 = Dense(units = num_classes, activation="softmax", name="output2")(flat_layer)
digit3 = Dense(units = num_classes, activation="softmax", name="output3")(flat_layer)
digit4 = Dense(units = num_classes, activation="softmax", name="output4")(flat_layer)
digit5 = Dense(units = num_classes, activation="softmax", name="output5")(flat_layer)

out_put = [digit1, digit2, digit3, digit4, digit5]

model = Model(inputs=input_shape, outputs=out_put)

model.summary()
model.load_weights(weights_file)
print('initialized')


optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



#Early stopping critera, model stops training when validation error becomes larger
checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_acc', verbose=1, save_best_only=True)

stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                              patience=1, min_lr=0.001)

tbCallBack = callbacks.TensorBoard(log_dir='checkpoints/custom_model',\
                                   histogram_freq=0, write_graph=True, write_images=True)

callbacks = [checkpoint, stopping, tbCallBack]


training_stats = model.fit(x=X_train, y=y_train2, validation_data=(X_val2, y_val2), epochs=7, \
                           batch_size=128, verbose=1, callbacks=callbacks)

model.save_weights(weights_file)
model.save("custom_model.h5")

# np.save('training_stats.npy', training_stats)
