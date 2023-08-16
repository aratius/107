import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from io import BytesIO
import numpy as np
import sys, os, base64, cv2
from PIL import Image
import math

def create_img(model):
  # トレーニング済みモデルを読み込んで画像を生成するプログラム
  random_latent_vectors = tf.random.normal(shape=(10, 128))
  generated_images = model.predict(random_latent_vectors)
  generated_images *= 255
  # PILのImageインスタンス
  img = keras.preprocessing.image.array_to_img(generated_images[0])
  return img

# モーフィングを作成
def create_morphing(model, z_dim, img_num_per_unit, morph_num):

  img_num = img_num_per_unit
  frame_num = img_num * (morph_num-1)
  zs = np.zeros((frame_num, z_dim))

  points = []
  for i in range(morph_num):
    points.append(np.clip(np.random.randn(z_dim), -1, 1))

  #間の補完
  #======================================
  for i in range(frame_num):
    this_i = int(math.floor(float(i) / float(img_num)))
    prev = points[this_i]
    crr = points[this_i+1]
    frame_this = i % img_num
    prev_ratio = (img_num-frame_this)/(img_num+1e-4)
    crr_ratio = frame_this/(img_num+1e-4)
    print(str(prev_ratio) + "," + str(crr_ratio))
    zs[i] = prev * prev_ratio + crr * crr_ratio
  #======================================

  #generatorで画像を生成
  #======================================
  imgs_array = model.predict(zs)
  imgs = []
  for i in range(len(imgs_array)):
      img = Image.fromarray(np.uint8(imgs_array[i] * 255))
      imgs.append(img)
  #======================================

  return imgs