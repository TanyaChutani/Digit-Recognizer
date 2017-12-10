import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

train = "train.csv"
#test = "../input/test.csv"
train = pd.read_csv(train)
#test = pd.read_csv(test)

print(train.isnull().any().sum())
#print(test.isnull().any().sum())

y= np.array(train["label"].values)
X= np.array(train.drop(labels = ["label"],axis = 1) )

#del train
#del test

x_train, x_v, y_train, y_v= train_test_split(X, y, test_size= 0.2, random_state= 14)

x_train= (x_train/255).reshape(-1, 28, 28, 1)

x_v= (x_v/255).reshape(-1, 28, 28, 1)

y_train =to_categorical(y_train)

y_v =to_categorical(y_v)

model= Sequential()
model.add(Conv2D(filters= 16, kernel_size= (5, 5), activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters= 32, kernel_size= (5, 5), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
datagen = ImageDataGenerator(zoom_range = 0.2, rotation_range = 10)


cb= LearningRateScheduler(lambda x: 0.001*0.7**x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=500,
                           epochs= 30, verbose=2, validation_data=(x_v[:200,:], y_v[:200,:]),
                           callbacks=[cb])
l, score= model.evaluate(x_v, y_v)
print("Loss: ", l)
print("Accuracy: ", score)

y_pred= model.predict(x_v)
y_pred = np.argmax(y_pred, axis=1)
y_v= np.argmax(y_v, axis=1)
cm= confusion_matrix(y_v, y_pred)
print(cm)