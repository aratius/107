import tensorflow as tf
from tensorflow import keras
import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/GAN")  # 下層のモジュールにはこの記述いらん説ある
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

model_path = input("model path ?")
output_dir = input("output dir ?")

os.makedirs(output_dir)

# モデルロード
model = tf.keras.models.load_model(model_path)

# トレーニング済みモデルを読み込んで画像を生成するプログラム
random_latent_vectors = tf.random.normal(shape=(10, 128))
generated_images = model.predict(random_latent_vectors)
generated_images *= 255
# generated_images.numpy()
for i in range(10):
    img = keras.preprocessing.image.array_to_img(generated_images[i])
    img.save(output_dir + "/generated_img_%d.png" % (i))

