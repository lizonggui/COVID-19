# -*- coding: utf-8 -*-
"""
@author: Zonggui Li
"""

import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import layers
from keras.layers import Input,Dense,Flatten,Embedding,Dropout,Reshape,LeakyReLU,Activation,UpSampling2D,Conv2D,Concatenate
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import backend as K
from InstanceNormalization  import InstanceNormalization

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

latent_dim = 112
height =112
width = 112
channels = 3
num_classes=3

#generator
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
n=Dense(256*14*14,activation='relu')(noise)
n=Reshape((14,14,256))(n)
l=Embedding(2,256)(label)
l=Dense(1*14*14,activation='relu')(l)
l=Reshape((14,14,1,))(l)
concat=Concatenate()([n,l])

x=UpSampling2D()(concat)
x=Conv2D(1024, kernel_size=7, padding="same")(x)
x=InstanceNormalization()(x)
x=Activation("relu")(x)
x1=cbam_block(x)

x=UpSampling2D()(x1)
x=Conv2D(512, kernel_size=7, padding="same")(x)
x=InstanceNormalization()(x)
x=Activation("relu")(x)
x2=cbam_block(x)


x=UpSampling2D()(x2)
x=Conv2D(256, kernel_size=7, padding="same")(x)
x=InstanceNormalization()(x)
x=Activation("relu")(x)
x3=cbam_block(x)

P1=UpSampling2D((4,4))(x1)
P2=UpSampling2D()(x2)

concat2=Concatenate()([P1,P2,x3])

x=Conv2D(3, kernel_size=7, padding="same")(x)
img=Activation("tanh")(x)
#
generator=Model([noise,label],img)
generator.summary()


#discriminator
img = Input(shape=(height,width,channels))

x =Conv2D(128,3,padding='same')(img)
x=LeakyReLU()(x)

x = Conv2D(128,4,strides=2,padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(128,4,strides=2,padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(128,4,strides=2,padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(128,4,strides=2,padding='same')(x)
x = LeakyReLU()(x)

x = Flatten()(x)

x = Dropout(0.5)(x)

validity = Dense(1, activation="sigmoid")(x)
label = Dense(num_classes, activation="softmax")(x)
discriminator=Model(img,[validity, label])
discriminator.summary()

noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
optimizer = Adam(lr=0.0002,
            clipvalue = 1.0,
            decay = 0.0001)

discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])
discriminator.trainable = False
valid, target_label = discriminator(img)
combined = Model([noise, label], [valid, target_label])
combined.summary()
combined.compile(loss=losses,
            optimizer=optimizer)

#train
epochs=40000
batch_size=4
sample_interval=10

def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    PATHS=tqdm([os.path.join(root,fn) for root,dirs,files in os.walk(DIR) for fn in files])
    print(PATHS)
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

NCP_train_label=np.ones(len(NCP_train))
Normal_train_label=np.full(len(Normal_train),-1)
CP_train_label = np.full(len(CP_train),2)

X_train =np.concatenate((NCP_train, Normal_train,CP_train), axis = 0)
Y_train = np.concatenate((NCP_train_label, Normal_train_label,CP_train_label), axis = 0)
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]
Y_train = to_categorical(Y_train, num_classes= 3)
        
X_train=X_train.reshape((X_train.shape[0],)+(112,112,3)).astype('float32')/255.
y_train = Y_train.reshape(-1, 1)

#adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

def save(model, model_name):
    model_path = "saved_model/%s.json" % model_name
    weights_path = "saved_model/%s_weights.hdf5" % model_name
    options = {"file_arch": model_path,
                        "file_weight": weights_path}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])


for epoch in range(epochs):
#             ---------------------
#              Train Discriminator
#             ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            sampled_labels = np.random.randint(0, 2, (batch_size, 1))

            # Generate  new images
            gen_imgs = generator.predict([noise, sampled_labels])

            img_labels = y_train[idx]
            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
#                save_model
                save(generator, "generator3")
                save(discriminator, "discriminator3")
#                sample_images
                r, c = 3,3
                noise = np.random.normal(0, 1, (r * c,latent_dim))
                sampled_labels = np.array([num for _ in range(r) for num in range(c)])
                gen_imgs = generator.predict([noise,sampled_labels])
#               Rescale images 0 - 1
                gen_imgs = 0.5 * gen_imgs + 0.5
                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                        axs[i,j].axis('off')
                        if cnt==0 or cnt==3 or cnt==6:
                            plt.imsave('generated_images/NCP/'+str(epoch)+str(cnt)+'.png',gen_imgs[cnt,:,:,:])
                        elif cnt==1 or cnt==4 or cnt==7:
                            plt.imsave('generated_images/Normal/'+str(epoch)+str(cnt)+'.png',gen_imgs[cnt,:,:,:])
                        elif cnt==2 or cnt==5 or cnt==8:
                            plt.imsave('generated_images/CP/'+str(epoch)+str(cnt)+'.png',gen_imgs[cnt,:,:,:])
                        cnt += 1
                plt.close()