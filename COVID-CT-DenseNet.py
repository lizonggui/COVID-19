# -*- coding: utf-8 -*-
"""
@author: Zonggui Li
"""

import numpy as np
import cv2
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os 
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K
from keras.applications import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score




#load data
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    PATHS=tqdm([os.path.join(root,fn) for root,dirs,files in os.walk(DIR) for fn in files])
    for PATH in PATHS:
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE,RESIZE))
            IMG.append(np.array(img))
    return IMG

NCP_train = np.array(Dataset_loader('./train/NCP',112))
Normal_train=np.array(Dataset_loader('./train/Normal',112))
CP_train=np.array(Dataset_loader('./train/CP',112))
NCP_val = np.array(Dataset_loader('./val/NCP',112))
Normal_val=np.array(Dataset_loader('./val/Normal',112))
CP_val=np.array(Dataset_loader('./val/CP',112))
NCP_test = np.array(Dataset_loader('./test/NCP',112))
Normal_test=np.array(Dataset_loader('./test/Normal',112))
CP_test=np.array(Dataset_loader('./test/CP',112))

NCP_train_label=np.ones(len(NCP_train))
Normal_train_label=np.zeros(len(Normal_train))
CP_train_label = np.full(len(CP_train),2)
NCP_val_label=np.ones(len(NCP_val))
Normal_val_label=np.zeros(len(Normal_val))
CP_val_label = np.full(len(CP_val),2)
NCP_test_label=np.ones(len(NCP_test))
Normal_test_label=np.zeros(len(Normal_test))
CP_test_label = np.full(len(CP_test),2)

X_train = np.concatenate((NCP_train, Normal_train,CP_train), axis = 0)
Y_train = np.concatenate((NCP_train_label, Normal_train_label,CP_train_label), axis = 0)
X_val=np.concatenate((NCP_val, Normal_val,CP_val), axis = 0)
Y_val = np.concatenate((NCP_val_label, Normal_val_label,CP_val_label), axis = 0)
X_test=np.concatenate((NCP_test, Normal_test,CP_test), axis = 0)
Y_test = np.concatenate((NCP_test_label, Normal_test_label,CP_test_label), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_val.shape[0])
np.random.shuffle(s)
X_val = X_val[s]
Y_val = Y_val[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

y_train = to_categorical(Y_train, num_classes=3)
y_val = to_categorical(Y_val, num_classes=3)
Y_test = to_categorical(Y_test, num_classes=3)

def label_smoothing(labels, epsilon=0.1):
        labels*=(1-epsilon)
        labels+=(epsilon/labels.shape[0])
        return labels
    
y_train = label_smoothing(y_train)
y_val=label_smoothing(y_val)
Y_test=label_smoothing(Y_test)


#attention(cbam_block)
def channel_attention(input_feature, ratio=7):
        channel = input_feature._keras_shape[-1]

        shared_layer_one = layers.Dense(channel // ratio,
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  use_bias=True,
                                  bias_initializer='zeros')
        shared_layer_two = layers.Dense(channel,
                                  kernel_initializer='he_normal',
                                  use_bias=True,
                                  bias_initializer='zeros')

        avg_pool = layers.GlobalAveragePooling2D()(input_feature)
        avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        max_pool = layers.GlobalMaxPooling2D()(input_feature)
        max_pool = layers.Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        cbam_feature = layers.Add()([avg_pool, max_pool])
        cbam_feature = layers.Activation('sigmoid')(cbam_feature)
        return layers.multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size=7):
        avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
        max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = layers.Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        return layers.multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=2):
        cbam_feature =channel_attention(cbam_feature, ratio)
        cbam_feature =spatial_attention(cbam_feature)
        return cbam_feature

#model
backbone=DenseNet201(weights='imagenet',
        include_top=False,
        input_shape=(112,112,3))

#Concat-FPN
P1=layers.Concatenate()([layers.MaxPooling2D(pool_size=(2, 2))(backbone.get_layer(name='conv1/relu').output),layers.Conv2D(64,1,padding='same')(backbone.get_layer(name='conv2_block6_concat').output)])
P2=layers.Concatenate()([layers.MaxPooling2D(pool_size=(2, 2))(backbone.get_layer(name='conv2_block6_concat').output),layers.Conv2D(256,1,padding='same')(backbone.get_layer(name='conv3_block12_concat').output)])
P3=layers.Concatenate()([layers.MaxPooling2D(pool_size=(2, 2))(backbone.get_layer(name='conv3_block12_concat').output),layers.Conv2D(512,1,padding='same')(backbone.get_layer(name='conv4_block32_concat').output)])
P4=layers.Concatenate()([layers.MaxPooling2D(pool_size=(2, 2))(backbone.get_layer(name='conv4_block32_concat').output),layers.Conv2D(1280,1,padding='same')(backbone.get_layer(name='conv5_block32_concat').output)])

P1=layers.Conv2D(128,5,strides=8,padding='valid')(P1)
P2=layers.Conv2D(128,3,strides=4,padding='valid')(P2)
P3=layers.Conv2D(128,3,strides=2,padding='valid')(P3)
P4=layers.Conv2D(128,3,strides=1,padding='same')(P4)

concat=layers.Concatenate()([P1,P2,P3,P4])
x=cbam_block(concat)
pool=layers.GlobalAveragePooling2D()(x)
bn=layers.BatchNormalization()(pool)
dense=layers.Dense(3,activation='softmax')(bn)
model=models.Model(input=backbone.inputs,output=dense)

#train
BATCH_SIZE =32
learning_rate=1e-4
filepath="weights.dataset1_COVID_CT_DenseNet.hdf5"

#data augmentation
train_generator = ImageDataGenerator(
        zoom_range=0.3,
        brightness_range=(0,0.3),
        horizontal_flip=True)

model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate,decay=0.0001),
        metrics=['accuracy'])

learn_control = ReduceLROnPlateau(monitor='val_acc', patience=1,
                                  verbose=1,factor=0.8, min_lr=1e-7)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
 
history = model.fit_generator(
    train_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[
            learn_control,
            checkpoint])

with open('history.json', 'w') as f:
    json.dump(str(history.history), f)
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()
plt.show()


'''
............................
*********test the trained model***********
............................

'''
#load the trained model
model=models.load_model("weights.dataset1_COVID_CT_DenseNet.hdf5")
Y_pred = model.predict(X_test)

#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('dataset1_COVID_CT_DenseNet_confusion_matrix.png',dpi=600)
    plt.show()

cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))

cm_plot_label =['NCP', 'Normal','CP']

plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Metrix of COVID_CT_DenseNet on Dataset1')


print('predict results:',np.argmax(Y_pred, axis=1))

print('groud-truth:',np.argmax(Y_test, axis=1))


#accuracy
accuracy=accuracy_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
#precision
precision=precision_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1),average='macro')
#recall
recall=recall_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1),average='macro')
#F1-score
F1_score=f1_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1),average='macro')

print('accuracy:',accuracy,'precision:',precision,'recall:',recall,'F1-score：',F1_score)