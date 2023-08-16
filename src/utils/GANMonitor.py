import tensorflow as tf
from tensorflow import keras
import sys, os
sys.path.append("C:\\Users\\Arata Matsumoto\\Documents\\git\\_Envs\\_Python\\FaceGan")
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

# GANの学習結果の生成画像を見るためのクラス
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128, proj_name="test"):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.proj_name = proj_name
        # 最初に保存先ディレクトリを作成
        os.makedirs(generated_path + "\\auto\\" + self.proj_name)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(generated_path + "\\auto\\" + self.proj_name + "\\%03d_%d.png" % (epoch, i))

