#!/usr/bin/env python
import cv2
import os
import rospy

from cv_bridge                  import CvBridge
from keras.models               import load_model
from sensor_msgs.msg            import Image

import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

img_size = 128
model = load_model('/home/rech/Documents/LESC/Petro2021/TecChallenge/modelo3MobileNet.h5')
# model.summary()

bridge = CvBridge()

def Predict(img):
    img_arr = img[...,::-1]
    resized_arr = cv2.resize(img_arr, (img_size, img_size))

    x = np.array(resized_arr)/255
    x.reshape(-1, img_size, img_size, 1)
    x = x[np.newaxis,...]
    
    predictions = model.predict(x, use_multiprocessing=True)
    
    # print(predictions)

    if predictions[0,0] > predictions[0,1]:
        print('Esta imagem não possui rachadura.')
    else:
        print('Esta imagem possui rachadura.')

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def image_callback(img_msg):
       
    cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    Predict(cv_image)

def listener():
    rospy.init_node('listener', anonymous=True)
    sub_image = rospy.Subscriber("/rech/img", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
