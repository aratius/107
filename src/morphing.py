import os, sys
import numpy as np
import numpy.random as nr
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
sys.path.append("C:\\Users\\Arata Matsumoto\\Documents\\git\\_Envs\\_Python\\FaceGan")
from src.utils.generate import create_morphing
from src.utils.config import train_model_path, generated_path, dataset_path, source_path

#パラメータ
#======================================
#使用する重み
param_reletional_path = input("param path : assets\\train_model\\gen\\[???]\\.h5 -> ")
param_path = "assets\\train_model\\gen\\" + param_reletional_path + ".h5"
#乱数列の次元
z_dim = 128
#モーフィングを保存するフォルダ
proj_name = input("proj name ? -> ")
proj_path = generated_path + "\\manual\\morph\\" + proj_name

#補完数（58だと60枚出ます）
morph_num = int(input("result num ? ->"))
point_num = int(input("point num (2 or higher) ? "))
img_num = int(input("img num per units ? ->"))
#======================================

#保存用フォルダ作成
if not os.path.isdir(proj_path):
    os.makedirs(proj_path)

def create_gif(index):

    # モデルのローディング
    model = load_model(param_path)

    # 画像の生成 画像配列が帰ってくる 外部関数
    imgs = create_morphing(model, z_dim, img_num, point_num)

    morph_path = proj_path + "\\morph_" + str(index)

    #保存用フォルダ作成
    if not os.path.isdir(morph_path):
        os.makedirs(morph_path)

    imgs_path = morph_path + "\\sources"
    if not os.path.isdir(imgs_path):
        os.makedirs(imgs_path)
    
    i = 0
    for img in imgs:
        img.save(imgs_path + "\\%d.png" % (i))
        i+=1

    #gif保存
    #======================================
    imgs[0].save(morph_path+"\\morph_.gif", save_all=True, append_images=imgs[1:], optimize=False, duration=1, loop=3)


for i in range(morph_num):
    create_gif(i)

os.system("start .\\" + proj_path)