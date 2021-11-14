import os, sys
import numpy as np
import numpy.random as nr
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
sys.path.append("/Users/matsumotoarata/git/ME/Python/GAN")  # 下層のモジュールにはこの記述いらん説ある
from src.utils.generate import create_morphing

#パラメータ
#======================================
#使用する重み
para = 'assets/train_model/gen/0913/0913gen_epoch400.h5'

#乱数列の次元
z_dim = 128
#補完数（58だと60枚出ます）
img_num = 58

#モーフィングを保存するフォルダ
img_f = 'assets/generated/1113_morphing_7/'
#======================================

def create_gif(index):

    model = load_model(para)

    imgs = create_morphing(model, z_dim, img_num)

    #保存用フォルダ作成
    if not os.path.isdir(img_f):
        os.makedirs(img_f)

    #gif保存
    #======================================
    imgs[0].save(img_f+'morph_'+str(index)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1, loop=3)


for i in range(10):
    create_gif(i)