import numpy as np
import os
import cv2
import pickle
import random

training_data = []
IMG_SIZE = 150
datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataSet", "train")
categories = ["NORMAL", "PNEUMONIA"]
def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    print(f"Fail to load image {img}")
                    continue
                img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_resize, class_num])
            except Exception as e:
                print(f"Error in loading image {img}: {e}")
create_training_data()

print(f"Total training data: {len(training_data)}")
random.shuffle(training_data)

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
pickle_out = open("y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()