import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from io import BytesIO
import numpy as np
import sys, os, base64, cv2
from PIL import Image

def create_img(model):
  # トレーニング済みモデルを読み込んで画像を生成するプログラム
  random_latent_vectors = tf.random.normal(shape=(10, 128))
  generated_images = model.predict(random_latent_vectors)
  generated_images *= 255
  # PILのImageインスタンス
  img = keras.preprocessing.image.array_to_img(generated_images[0])
  # base64エンコーディング
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  print(img_str)

# モーフィングを作成
def create_morphing(model, z_dim, img_num, ):
  #乱数列１
  a = np.clip(np.random.randn(z_dim), -1, 1)
  print(a)
  #乱数列2
  b = np.clip(np.random.randn(z_dim), -1, 1)
  print(b)

  #間の補完
  #======================================
  zs = np.zeros((img_num+2, z_dim))
  for i in range(img_num+2):
      zs[i] = a * ((img_num+1-i)/(img_num+1)) + b * (i/(img_num+1))
  #======================================

  #generatorで画像を生成
  #======================================
  model = load_model(para)
  imgs_array = model.predict(zs)
  imgs = []
  for i in range(len(imgs_array)):
      img = Image.fromarray(np.uint8(imgs_array[i] * 255))
      imgs.append(img)
  #======================================

  return imgs[0]