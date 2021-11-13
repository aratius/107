import os, sys
import numpy as np
import numpy.random as nr
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
sys.path.append("/Users/matsumotoarata/git/ME/Python/GAN")  # 下層のモジュールにはこの記述いらん説ある

#パラメータ
#======================================
#使用する重み
para = 'assets/train_model/gen/0913/0913gen_epoch400.h5'

#乱数列の次元
z_dim = 128
#補完数（58だと60枚出ます）
img_num = 58

#モーフィングを保存するフォルダ
img_f = 'assets/generated/1113_morphing_4/'
#======================================

def create_gif(index):
    #乱数列１
    a = np.clip(nr.randn(z_dim), -1, 1)
    print(a)
    #乱数列2
    b = np.clip(nr.randn(z_dim), -1, 1)
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
        img = keras.preprocessing.image.array_to_img(imgs_array[i])
        # img = Image.fromarray(np.uint8(imgs_array[i] * 127.5 + 127.5))
        imgs.append(img)
    #======================================

    print("hello ------------------------------------")

    #保存用フォルダ作成
    if not os.path.isdir(img_f):
        os.makedirs(img_f)

    #保存
    #======================================
    # for i in range(len(imgs)):
    #     print(i)
        #画像の表示と保存
        # imgs[i].save(img_f + str(i) + '.png')
        # plt.imshow(imgs[i], vmin = 0, vmax = 255)
        # plt.show()
    #gif保存
    #======================================
    imgs[0].save(img_f+'morph_'+str(index)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1, loop=0)


for i in range(10):
    create_gif(i)