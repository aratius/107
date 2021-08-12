# https://keras.io/examples/generative/dcgan_overriding_train_step/
# 顔画像自動生成

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/GAN")
from src.utils.GAN import GAN
from src.utils.GANMonitor import GANMonitor
from src.utils.modelSaver import ModelSaver
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

# 環境ごとのユニークな値を最初に入力させる
project_name = input("proj name ?")
dataset_name = input("dataset dir name ?")

# データセットを作成、値を正規化
dataset = keras.preprocessing.image_dataset_from_directory(
  dataset_path + dataset_name, label_mode=None, image_size=(128,128), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)

# 鑑定者のニューラルネットワーク
discriminator = keras.Sequential(
  [
    # 64x64px, rgb 3channel
    keras.Input(shape=(128, 128, 3)),
    # 畳み込み層
    layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
    # 活性化関数
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dropout(0.2),
    # 全結合層 出力
    layers.Dense(1, activation="sigmoid")
  ],
  name="discriminator"
)
# 正常に動くかチェック
discriminator.summary()

# 贋作者のニューラルネットワーク
latent_dim = 128
generator= keras.Sequential(
  [
    keras.Input(shape=(latent_dim,)),
    layers.Dense(16 * 16 * 128),
    layers.Reshape((16, 16, 128)),
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
    layers.ReLU(),  # ReLUにした
    layers.BatchNormalization(),
    layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
    layers.ReLU(),  # ReLUにした
    layers.BatchNormalization(),
    layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
    layers.ReLU(),  # ReLUにした
    # layers.BatchNormalization(),
    layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
  ],
  name="generator"
)
# 正常に動くかチェック
generator.summary()

epochs = 200  # In practice, use ~100 epochs
# GANをインスタンス化
# discriminator.trainable = False
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
# 最適化手法を設定
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.1),  # beta_1てなに
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),  # を強くする
    loss_fn=keras.losses.BinaryCrossentropy(),
)

# 最初にモデル保存テスト
generator.save(train_model_path + "gen/" + project_name + "/test.h5")
discriminator.save(train_model_path + "disc/" + project_name + "/test.h5")

# 学習
gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim, proj_name=project_name), ModelSaver(generator, discriminator)]
)

# 最後にもモデル保存
generator.save(train_model_path + "gen/" +  + project_name + "/last.h5")
discriminator.save(train_model_path + "disc/" +  + project_name + "/last.h5")

print("hoge")