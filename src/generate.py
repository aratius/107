import tensorflow as tf
from tensorflow import keras
from io import BytesIO
import numpy as np
import sys, os, base64, cv2
sys.path.append("C:\\Users\\Arata Matsumoto\\Documents\\git\\_Envs\\_Python\\FaceGan")
from src.utils.config import train_model_path, generated_path, dataset_path, source_path
from src.utils.generate import create_img

model_reletional_path = input("model_path : assets\\train_model\\gen\\[???]\\.h5 -> ")
model_path = "assets\\train_model\\gen\\" + model_reletional_path + ".h5"

proj_name = input("proj name? -> ")
proj_path = generated_path + "\\manual\\static\\" + proj_name

img_num = int(input("img num ? -> "))

#保存用フォルダ作成
if not os.path.isdir(proj_path):
    os.makedirs(proj_path)

# モデルロード
model = tf.keras.models.load_model(model_path)

img = create_img(model)
for i in range(img_num):
    img.save(proj_path + "\\%d.png" % (i))