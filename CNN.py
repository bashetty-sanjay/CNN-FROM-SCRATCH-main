# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from itertools import produc
# x=[24,25,26]
# y=[23,24,25]
# plt.plot(x,y)
# plt.show()
# a=np.array([1,2,3,4,5,6])
# x=np.arange(1,28,1)
# print(x)
# y = x.reshape (3,3,3)
# print(y)
# print(y[1:3,1:3,1:3])
# a=np.zeros(5)
# b=np.ones(5)
# print(a)

# array2d = np.array([
# [1,1,1,1,1,1,1,1,1],
# [1,1,2,1,1,1,2,1,1], 
# [1,2,2,2,1,2,2,2,1],
# [2,2,2,2,2,2,2,2,2],
# [1,2,2,2,2,2,2,2,1], 
# [1,1,2,2,2,2,2,1,1],
# [1,1,1,2,2,2,1,1,1],
# [1,1,1,1,2,1,1,1,1],
# ])
# plt.imshow(array2d, cmap = "gray")
# plt.show()
# # array2d.shape
# # arr2d = np.random.randint(2, 10, size=(10, 10))
# # plt.imshow(arr2d, cmap="gray")
# # plt.show()
# # arr2d.shape
# # print (arr2d)
# # arr3d = np.random.randint(2, 255, size=(5, 5, 1))
# # plt.imshow(arr2d, cmap= "gray")
# # plt.show()
# # arr2d.shape
# arr3d=np.random.randint(2,255,size=(5,5,3))
# arr3d_copy1= arr3d.copy()
# arr3d_copy1[:,:,1]=2
# arr3d_copy1[:,:,2]=2
# plt.figure(figsize=(3,3))
# plt.imshow(arr3d_copy1)
# plt.show()
# my_image = plt.imread(r"C:\Users\Mahesh\Downloads\Telegram Desktop\316666.jpg")
# plt.imshow(my_image[::-1])
# plt.show()
# print(my_image)
# print(my_image[1:3,1:3,1:3])






import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D

DATADIR = r"D:\Codeeee\Project"
CATEGORIES = ["NORMAL_TRAIN", "PNEUMONIA_TRAIN"]
IMG_SIZE = 150
training_data = []
def training():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        if os.path.exists(path):
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                plt.imshow(img_array, cmap='gray')
                # plt.show()
                plt.imshow(img_resize, cmap='gray')
                # plt.show()
                training_data.append([img_resize,class_num])  
                # break
            # break
training()
import random
random.shuffle(training_data)
print(len(training_data))
for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for feature, label in training_data:
	x.append(feature)
	y.append(label)

X = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

# print(X.shape, Y.shape)
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, epochs=2, validation_split=0.02)












# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
# # more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
# from tensorflow.keras.callbacks import TensorBoard
# import pickle
# import time

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

# X = X/255.0

# dense_layers = [0]
# layer_sizes = [64]
# conv_layers = [3]

# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#             print(NAME)

#             model = Sequential()

#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation('relu'))
#             model.add(MaxPooling2D(pool_size=(2, 2)))

#             for l in range(conv_layer-1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))

#             model.add(Flatten())

#             for _ in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation('relu'))

#             model.add(Dense(1))
#             model.add(Activation('sigmoid'))

#             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#             model.compile(loss='binary_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'],
#                           )

#             model.fit(X, y,
#                       batch_size=32,
#                       epochs=10,
#                       validation_split=0.3,
#                       callbacks=[tensorboard])

# model.save('64x3-CNN.model')