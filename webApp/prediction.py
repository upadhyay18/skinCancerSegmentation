# import all required libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# loss functions
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

## load model
def load_model():
    from tensorflow.keras.utils import CustomObjectScope
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("") #<<<<---------- Add path here
    return model

# function to read images and save result

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (256, 256))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def save_results(ori_x, y_pred,x, save_image_path):
    comb_img=cv2.bitwise_and(ori_x,ori_x,mask=y_pred)
    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)
    cv2.imwrite(save_image_path+"_output.jpg", y_pred*255)
    cv2.imwrite(save_image_path+"_output2.jpg", comb_img)
    cv2.imwrite(save_image_path+".jpg", ori_x)

def predict(path):
    name=path.split("/")[-1].split(".")[0]
    x=path
    model=load_model()
    ori_x, x = read_image(x) # read image
    # make prediciton
    y_pred = model.predict(x)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.uint8)
    save_image_path = os.getcwd()+f"/static/projectImg/{name}"
    save_results(ori_x, y_pred,x, save_image_path)

