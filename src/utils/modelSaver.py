import tensorflow as tf
from tensorflow import keras
# import sys
# sys.path.append("/Users/matsumotoarata/git/ME/Python/GAN")  # 下層のモジュールにはこの記述いらん説ある
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

# モデルを保存する
class ModelSaver(keras.callbacks.Callback):
    def __init__(self, gen_model, disc_model, proj_name):
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.proj_name = proj_name

    def on_epoch_end(self, epoch, logs=None):
      # モデルの途中経過を保存
      if(epoch % 10 == 0):
        self.gen_model.save(train_model_path + "gen\\" + self.proj_name + "\\gen_epoch" + str(epoch+1) + ".h5")
        self.disc_model.save(train_model_path + "disc\\" + self.proj_name + "\\disc_epoch" + str(epoch+1) + ".h5")
