import os

import cv2                  as cv
import tensorflow           as tf
import numpy                as np

from sklearn.model_selection    import train_test_split
from sklearn.metrics            import classification_report

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

labels = ['negativo', 'positivo']
img_size = 128

def get_data(data_dir):
    data = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)        
        for img in os.listdir(path):
            try:
                img_arr = cv.imread(os.path.join(path, img))[...,::-1] 
                resized_arr = cv.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

data = get_data('/home/rech/Documents/Datamining/trabalho/CrackClassification')

x = []
y = []

for feature, label in data:
  x.append(feature)
  y.append(label)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

X_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

X_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

base_model = tf.keras.applications.MobileNetV2(input_shape = (img_size, img_size, 3), include_top = False, weights = "imagenet")


base_model.trainable = False

model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(2, activation="softmax")])

model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs = 150 , validation_data = (X_test, y_test))

model.save("/home/rech/Documents/LESC/Petro2021/TecChallenge/modelo3MobileNet.h5")

from time import time 

print(X_test.shape)

t0 = time()
predictions = model.predict(X_test[:1], use_multiprocessing=True)
t1 = time()

pred_aux = []

for i in range(predictions.shape[0]):
    if predictions[i , 0] > predictions[i , 1]:
        pred_aux.append(0)
    else:
        pred_aux.append(1)

from sklearn import metrics

# cm = metrics.confusion_matrix(y_test, pred_aux)
# print()
# print(cm)

# print(classification_report(y_test, pred_aux, target_names = ['negativo (Class 0)','positivo (Class 1)']))
# print ('function vers1 takes %f segundos' %(t1-t0))