import tensorflow as tf
from tensorflow import keras
from io import BytesIO
import numpy as np
import sys, os, base64, cv2
sys.path.append("/Users/aualrxse/git/ME/FaceGan")  # 下層のモジュールにはこの記述いらん説ある
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

model_path = "assets/train_model/test/gen_epoch181.h5"

# モデルロード
model = tf.keras.models.load_model(model_path)

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