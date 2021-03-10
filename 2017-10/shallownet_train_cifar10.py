from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.utils import np_utils
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD,RMSprop,adam
import os


class ShallowCNet():
    @staticmethod
    def build(width, height, depth, classes,weights_path=None):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        #model = Sequential()
        inputShape = (width, height, depth)

        # first CONV => RELU
        In_=Input(inputShape)
        c1=Conv2D(32, (3, 3), padding="same", input_shape=inputShape)(In_)
        a1=Activation("relu")(c1)

        # first CONV => RELU => POOL
        c2=Conv2D(32, (3, 3), padding="same")(a1)
        a2=Activation("relu")(c2)
        p1=MaxPooling2D(pool_size=(2, 2))(a2)

        # first (and only) set of FC => RELU => Dropout layers
        d1=Flatten()(p1)
        d2=Dense(512)(d1)
        d3=Activation("relu")(d2)
        d4=Dropout(0.5)(d3)

        # softmax classifier
        output=Dense(classes, activation='softmax')(d4)
        model=Model(inputs=In_,outputs=output)

        if weights_path:
            model.load_weights(weights_path)

        return model




print("[INFO] loading CIFAR10 data")

((trainX,trainY),(testX,testY))=cifar10.load_data()

trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0

print(trainX.shape)
print(testX.shape)

np.random.seed(123)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(trainY, 10)
Y_test = np_utils.to_categorical(testY, 10)
print(Y_test[0:5])
print(Y_test.argmax(axis=1))

labelNames=["airplane","automobile","bird","cat","deer","dog",
            "frog","horse","ship","truck"]
#
# # initialize the optimizer and model
print("[INFO] compiling model...")
#opt = SGD(lr=0.01, decay=0.01 / 10, momentum=0.9, nesterov=True)
opt = adam(lr=0.001, decay=0.01/10)
model = ShallowCNet.build(width=32, height=32, depth=3, classes=10,weights_path=None)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
nb_epoch=10
no_epochs=nb_epoch

# train the network
print("[INFO] training network...")
H = model.fit(trainX, Y_train, validation_data=(testX, Y_test),batch_size=64,
epochs=no_epochs,verbose=1)

weights_dir='cifar'

if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# save model
model.save_weights(os.path.join(weights_dir,'cifar_train_weights.h5'))

import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, no_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, no_epochs), H.history["val_loss"], label="val_loss")
plt.title('Loss history')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, no_epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, no_epochs), H.history["val_acc"], label="val_acc")
plt.title('Classification accuracy history')
plt.xlabel('Number of Epochs')
plt.ylabel('Clasification accuracy')
plt.grid(True)
plt.show()
#
