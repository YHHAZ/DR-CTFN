from __future__ import absolute_import, print_function

import json
import math
import os
import shutil
import sys

# 删除文件，然后移动文件
os.remove('/home/luyaping_w/project/software/Anaconda/lib/python3.9/site-packages/keras/preprocessing/image.py')
shutil.copy('/home/luyaping_w/project/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/0_数据预处理对比/Fusion/TestAll/image.py',
            '/home/luyaping_w/project/software/Anaconda/lib/python3.9/site-packages/keras/preprocessing/')


import keras
import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, Lambda,
                                     LeakyReLU, Multiply, Permute, ReLU,
                                     Reshape, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from model1 import ConvNeXtBase, LayerScale
from model2 import SwinTransformerLarge384

gpus = tf.config.experimental.list_physical_devices(
    'GPU')  #如果设置gpu可见的话，默认是所有gpu都是可用的，所以这个程序使用4个gpu同时可用的状态
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPU的个数")
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    class LayerScale(layers.Layer):
        """Layer scale module.

        References:
        - https://arxiv.org/abs/2103.17239

        Args:
        init_values (float): Initial value for layer scale. Should be within
            [0, 1].
        projection_dim (int): Projection dimensionality.

        Returns:
        Tensor multiplied to the scale.
        """

        def __init__(self, init_values, projection_dim, **kwargs):
            super().__init__(**kwargs)
            self.init_values = init_values
            self.projection_dim = projection_dim

        def build(self, input_shape):
            self.gamma = self.add_weight(
                name="gamma",
                shape=(self.projection_dim,),
                initializer=initializers.Constant(self.init_values),
                trainable=True,
            )

        def call(self, x):
            return x * self.gamma

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "init_values": self.init_values,
                    "projection_dim": self.projection_dim,
                }
            )
            return config
        
    def LightAttentionBlock(inputs,b=1, gamma=2,name=''):
        # 通道注意力机制
        channels = inputs.shape[-1]
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        x_global_avg_pool = GlobalAveragePooling2D(name=name+'GlobalAveragePooling2D')(inputs)
        x = Reshape((channels,1),name=name+'Reshape')(x_global_avg_pool)
        x = Conv1D(1, kernel_size=k,padding="same",name=name+'Conv1D')(x)
        x = Activation('sigmoid',name=name+'Activation')(x)
        x = Reshape((1, 1, channels),name=name+'Reshape2')(x)
        output = Multiply(name=name+'Multiply')([inputs,x])

        # 空间注意力机制
        x = tf.reduce_mean(output,axis=-1,keepdims=True,name=name+'reduce_mean')
        x = Activation('sigmoid',name=name+'Activation2')(x)
        output = Multiply(name=name+'Multiply2')([x, output])
        return output

    
    def ConvNeXtBase_build_model(inputs_dim,num_classes,
                                input_shape=(512, 512, 3)):
        x_0,x_1,x_2,x_3,x = ConvNeXtBase(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)(inputs_dim)

        x = LightAttentionBlock(x,name='build_model_')
        x = GlobalAveragePooling2D(
            name='build_model_ConvNeXtBase_main_GlobalAveragePooling2D')(x)
        dp_1 = Dropout(0.5, name='build_model_ConvNeXtBase_main_Dropout1')(x)
        fc2_num_classes = Dense(
            512,
            kernel_initializer='he_normal',
            name='build_model_ConvNeXtBase_main_Dense_3')(dp_1)
        dp_2 = Dropout(0.5, name='build_model_ConvNeXtBase_main_Dropout2')(fc2_num_classes)
        fc2_num_classes2 = Dense(
            num_classes,
            kernel_initializer='he_normal',
            name='build_model_ConvNeXtBase_main_Dense_4')(dp_2)
        fc2_num_classes2 = Activation(
            'softmax', name='build_model_ConvNeXtBase')(fc2_num_classes2)
        model = Model(inputs=inputs_dim, outputs=[x_0,x_1,x_2,x_3,fc2_num_classes2])
        model.load_weights('/home/luyaping_w/project/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/0_数据预处理对比/Fusion/Test/model_1_weights.h5')
        
        model = Model(inputs=model.input, 
                  outputs=[model.output[0],model.output[1],model.output[2],model.output[3],model.get_layer('build_model_ConvNeXtBase_main_GlobalAveragePooling2D').output])
        return model
    
    def SwinTransformerLarge384_build_model(inputs_dim,num_classes,
                                input_shape=(512, 512, 3)):
        x_0,x_1,x_2,x_3,x = SwinTransformerLarge384(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)(inputs_dim)

        x = LightAttentionBlock(x,name='build_model_')
        x = GlobalAveragePooling2D(
            name='build_model_SwinTransformerLarge384_main_GlobalAveragePooling2D')(x)
        dp_1 = Dropout(0.5, name='build_model_SwinTransformerLarge384_main_Dropout1')(x)
        fc2_num_classes = Dense(
            512,
            kernel_initializer='he_normal',
            name='build_model_SwinTransformerLarge384_main_Dense_3')(dp_1)
        dp_2 = Dropout(0.5, name='build_model_SwinTransformerLarge384_main_Dropout2')(fc2_num_classes)
        fc2_num_classes2 = Dense(
            num_classes,
            kernel_initializer='he_normal',
            name='build_model_SwinTransformerLarge384_main_Dense_4')(dp_2)
        fc2_num_classes2 = Activation(
            'softmax', name='build_model_SwinTransformerLarge384')(fc2_num_classes2)
        model = Model(inputs=inputs_dim, outputs=[x_0,x_1,x_2,x_3,fc2_num_classes2])
        model.load_weights('/home/luyaping_w/project/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/0_数据预处理对比/Fusion/Test/model_2_weights.h5',by_name=True, skip_mismatch=True)
        
        model = Model(inputs=model.input, 
                  outputs=[model.output[0],model.output[1],model.output[2],model.output[3],model.get_layer('build_model_SwinTransformerLarge384_main_GlobalAveragePooling2D').output])
        return model
    
    def build_model(num_classes, input_shape):
        inputs_dim = Input(input_shape)
        ConvNeXtBase_0,ConvNeXtBase_1,ConvNeXtBase_2,ConvNeXtBase_3,ConvNeXtBase_num_classes = ConvNeXtBase_build_model(
            inputs_dim, num_classes, (input_shape[0], input_shape[1], input_shape[2]))(inputs_dim)
        SwinTransformerLarge384_0,SwinTransformerLarge384_1,SwinTransformerLarge384_2,SwinTransformerLarge384_3, SwinTransformerLarge384_0_num_classes = SwinTransformerLarge384_build_model(
            inputs_dim, num_classes, (input_shape[0], input_shape[1], input_shape[2]))(inputs_dim)

        # 第0层的拼接
        output_0 = Concatenate(axis=-1,name='build_model_Concatenate_{}'.format(str(0)))([ConvNeXtBase_0,SwinTransformerLarge384_0])
        sq0 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='he_normal',
                          name='build_model_conv2D_{}'.format(str(0))),
            BatchNormalization(name='build_model_BatchNormalization_{}'.format(str(0))),
            ReLU(name='build_model_ReLU_{}'.format(str(0))),
            GlobalAveragePooling2D(name='build_model_main_GlobalAveragePooling2D_'+str(0)),
            Dropout(0.6, name='build_model_main_Dropout_'+str(0)),
            Dense(num_classes,kernel_initializer='he_normal',name='build_model_main_Dense_1_'+str(0)),
        ],name='build_model_main_sq0')(output_0)
        sq0 = Activation('softmax',name='build_model_main_softmax_'+str(0))(sq0)
        # 第1层的拼接
        output_1 = Concatenate(axis=-1,name='build_model_Concatenate_{}'.format(str(1)))([ConvNeXtBase_1,SwinTransformerLarge384_1])
        sq1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='he_normal',
                          name='build_model_conv2D_{}'.format(str(1))),
            BatchNormalization(name='build_model_BatchNormalization_{}'.format(str(1))),
            ReLU(name='build_model_ReLU_{}'.format(str(1))),
            GlobalAveragePooling2D(name='build_model_main_GlobalAveragePooling2D_'+str(1)),
            Dropout(0.6, name='build_model_main_Dropout_'+str(1)),
            Dense(num_classes,kernel_initializer='he_normal',name='build_model_main_Dense_1_'+str(1)),
        ],name='build_model_main_sq1')(output_1)
        sq1 = Activation('softmax',name='build_model_main_softmax_'+str(1))(sq1)
        # 第2层的拼接
        output_2 = Concatenate(axis=-1,name='build_model_Concatenate_{}'.format(str(2)))([ConvNeXtBase_2,SwinTransformerLarge384_2])
        sq2 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='he_normal',
                          name='build_model_conv2D_{}'.format(str(2))),
            BatchNormalization(name='build_model_BatchNormalization_{}'.format(str(2))),
            ReLU(name='build_model_ReLU_{}'.format(str(2))),
            GlobalAveragePooling2D(name='build_model_main_GlobalAveragePooling2D_'+str(2)),
            Dropout(0.6, name='build_model_main_Dropout_'+str(2)),
            Dense(num_classes,kernel_initializer='he_normal',name='build_model_main_Dense_1_'+str(2))
        ],name='build_model_main_sq2')(output_2)
        sq2 = Activation('softmax',name='build_model_main_softmax_'+str(2))(sq2)
        
        # 第3层的拼接
        output_3 = Concatenate(axis=-1,name='build_model_Concatenate_{}'.format(str(3)))([ConvNeXtBase_3,SwinTransformerLarge384_3])
        sq3 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer='he_normal',
                          name='build_model_conv2D_{}'.format(str(3))),
            BatchNormalization(name='build_model_BatchNormalization_{}'.format(str(3))),
            ReLU(name='build_model_ReLU_{}'.format(str(3))),
            GlobalAveragePooling2D(name='build_model_main_GlobalAveragePooling2D_'+str(3)),
            Dropout(0.6, name='build_model_main_Dropout_'+str(3)),
            Dense(num_classes,kernel_initializer='he_normal',name='build_model_main_Dense_1_'+str(3)),
        ],name='build_model_main_sq')(output_3)
        sq3 = Activation('softmax',name='build_model_main_softmax_{}'.format(str(3)))(sq3)
        # label的拼接
        model = Concatenate(name='build_model_main_Concatenate')([
            ConvNeXtBase_num_classes, SwinTransformerLarge384_0_num_classes
        ])
        model = Dropout(0.6, name='build_model_main_Dropout_1')(model)
        model = Dense(512,
                      kernel_initializer='he_normal',
                      name='build_model_main_Dense_1')(model)
        model = LeakyReLU(alpha=0.0001,
                        name='build_model_main_LeakyReLU')(
                            model)  #此处注意，为sigmoid函数
        model = Dropout(0.6, name='build_model_main_Dropout_2')(model)
        model = Dense(num_classes,
                      kernel_initializer='he_normal',
                      name='build_model_main_Dense_2')(model)
        All_num_classes = Activation('softmax',
                                     name='build_model_All_num_classes')(
                                         model)  #此处注意，为sigmoid函数
        model = Model(inputs=inputs_dim,
                      outputs=[sq0,sq1,sq2,sq3,All_num_classes])
        return model

    def setup_to_fine_tune_1(model):
        LayersNum = 0
        for layer in model.layers:
            if not layer.name.startswith('build_model'):
                layer.trainable = False
                LayersNum += 1
        print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " +
              str(len(model.layers) - LayersNum))

        loss={'build_model_main_softmax_0': 'categorical_crossentropy',
              'build_model_main_softmax_1': 'categorical_crossentropy',
              'build_model_main_softmax_2': 'categorical_crossentropy',
              'build_model_main_softmax_3': 'categorical_crossentropy',
              'build_model_All_num_classes': 'categorical_crossentropy'}
        loss_weights={'build_model_main_softmax_0': 1,
                      'build_model_main_softmax_1': 1,
                      'build_model_main_softmax_2': 1,
                      'build_model_main_softmax_3': 1,
                      'build_model_All_num_classes': 1}


        model.compile(optimizer=Adam(lr=0.01),
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=['accuracy'])

    def setup_to_fine_tune_2(model):
        LayersNum = 0
        for layer in model.layers:
            layer.trainable = True
            LayersNum += 1
        print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " +
              str(len(model.layers) - LayersNum))
        loss={'build_model_main_softmax_0': 'categorical_crossentropy',
              'build_model_main_softmax_1': 'categorical_crossentropy',
              'build_model_main_softmax_2': 'categorical_crossentropy',
              'build_model_main_softmax_3': 'categorical_crossentropy',
              'build_model_All_num_classes': 'categorical_crossentropy'}
        loss_weights={'build_model_main_softmax_0': 1,
                      'build_model_main_softmax_1': 1,
                      'build_model_main_softmax_2': 1,
                      'build_model_main_softmax_3': 1,
                      'build_model_All_num_classes': 1}
        model.compile(optimizer=Adam(lr=0.000001),
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=['accuracy'])

    # height = 256
    # width = 256
    # channels = 3
    # batch_size = 16*4
    # num_classes = 5
    # SEED = 666
    # epochs = 300

    # train_dir="/project/luyaping_w/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/Datasets/5_DRPixelDatasets"
    # valid_dir="/project/luyaping_w/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/Datasets/3_DRTestDatasetsProcess"


    # train_datagen = keras.preprocessing.image.ImageDataGenerator()  # 底层已经做归一化了

    # train_generator = train_datagen.flow_from_directory(
    #     train_dir,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
    #     target_size=(height, width),  #图片大小规定到这个高宽
    #     batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
    #     seed=SEED,
    #     shuffle=True,
    #     class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
    # valid_datagen = keras.preprocessing.image.ImageDataGenerator()  #验证集不用添加图片，只需要将图片像素值进行规定
    # valid_generator = valid_datagen.flow_from_directory(
    #     valid_dir,
    #     target_size=(height, width),
    #     batch_size=batch_size,
    #     seed=SEED,
    #     shuffle=False,
    #     class_mode="categorical")
    # train_num = train_generator.samples  #获取训练样本总数
    # valid_num = valid_generator.samples
    # print("样本总数为：")
    # print(train_num, valid_num)

    # input_shape = (height, width, channels)
    # model = build_model(num_classes,input_shape)
    # setup_to_fine_tune_1(model)
    # model.summary()
    
    # import os
    # output_model_file = r'./callbacks_EarlyStopping_1'
    # if not os.path.exists(output_model_file):
    #     os.mkdir(output_model_file)
    # output_model_file = output_model_file + '/callbacks_EarlyStopping.h5'
    # log_dir = os.path.join('log_1')  #win10下的bug，
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)

    # #回调函数的使用-在训练中数据的保存
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath = output_model_file,  #最后模型的保存-加上下面的代码代表就是最优模型的保存
    #         monitor='val_loss',
    #         save_best_only=True,save_weights_only=True),
    #     keras.callbacks.EarlyStopping(
    #         monitor='val_loss', mode='auto', min_delta=1e-10, patience=13
    #     ),  #如果模型提前关闭的参数设置，patience参数的意义在于:当迭代次数5次检测指标的值都是比我规定的是小的话，就直接停止模型的训练
    #     #min_delta参数的意思就是:本次训练的测试指标的值与上一次的值的差值是不是比这个阈值要低，如果低的话就停止模型的训练
    #     keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
    #                                       patience=5,
    #                                       mode='auto',
    #                                       verbose=1,
    #                                       min_delta=1e-9),
    #     tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    # ]

    # history = model.fit_generator(
    #     train_generator,  #迭代器对象
    #     steps_per_epoch=train_num // batch_size,  #因为迭代器是无限次的，所以要规定什么时候退出
    #     epochs=epochs,
    #     validation_data=valid_generator,
    #     validation_steps=valid_num // batch_size,
    #     callbacks=callbacks)
    # print('Saving model to disk\n')
    # model.save_weights('model_Deep_ensemble_learning_1.h5')
    # print("history保存")
    # import pickle
    # with open('model_Deep_ensemble_learning_1.pickle', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    height = 256
    width = 256
    channels = 3
    batch_size = 4*4
    num_classes = 5
    SEED = 666
    epochs = 300
    
    train_dir="/project/luyaping_w/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/Datasets/5_DRPixelDatasets"
    valid_dir="/project/luyaping_w/code/EyeDR/DeepEnsembleLearning/SwinTransformerConvNeXt/Datasets/3_DRTestDatasetsProcess"

    train_datagen = keras.preprocessing.image.ImageDataGenerator()  # 底层已经做归一化了

    train_generator = train_datagen.flow_from_directory(
        train_dir,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
        target_size=(height, width),  #图片大小规定到这个高宽
        batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
        seed=SEED,
        shuffle=True,
        class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
    valid_datagen = keras.preprocessing.image.ImageDataGenerator()  #验证集不用添加图片，只需要将图片像素值进行规定
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(height, width),
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode="categorical")
    train_num = train_generator.samples  #获取训练样本总数
    valid_num = valid_generator.samples
    print("样本总数为：")
    print(train_num, valid_num)

    input_shape = (height, width, channels)
    model = build_model(num_classes,input_shape)
    model.load_weights(r'/home/luyaping_w/scratch/callbacks_EarlyStopping_3/callbacks_EarlyStopping_13_0.8179.h5')
    setup_to_fine_tune_2(model)
    model.summary()
    
    import os
    output_model_file = r'./callbacks_EarlyStopping_3'
    if not os.path.exists(output_model_file):
        os.mkdir(output_model_file)
    output_model_file = output_model_file + '/callbacks_EarlyStopping_{epoch:02d}_{val_build_model_All_num_classes_accuracy:.4f}.h5'
    log_dir = os.path.join('log_3')  #win10下的bug，
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #回调函数的使用-在训练中数据的保存
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = output_model_file,  #最后模型的保存-加上下面的代码代表就是最优模型的保存
            monitor='val_build_model_All_num_classes_accuracy',
            save_best_only=False,save_weights_only=True),
        keras.callbacks.EarlyStopping(
            monitor='val_build_model_All_num_classes_accuracy', mode='auto', min_delta=1e-10, patience=13
        ),  #如果模型提前关闭的参数设置，patience参数的意义在于:当迭代次数5次检测指标的值都是比我规定的是小的话，就直接停止模型的训练
        #min_delta参数的意思就是:本次训练的测试指标的值与上一次的值的差值是不是比这个阈值要低，如果低的话就停止模型的训练
        keras.callbacks.ReduceLROnPlateau(monitor='val_build_model_All_num_classes_accuracy',
                                          patience=5,
                                          mode='auto',
                                          verbose=1,
                                          min_delta=1e-9),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit_generator(
        train_generator,  #迭代器对象
        # steps_per_epoch=train_num // batch_size,  #因为迭代器是无限次的，所以要规定什么时候退出
        epochs=epochs,
        validation_data=valid_generator,
        # validation_steps=valid_num // batch_size,
        callbacks=callbacks)
    print('Saving model to disk\n')
    model.save_weights('model_Deep_ensemble_learning_2.h5')
    print("history保存")
    import pickle
    with open('model_Deep_ensemble_learning_2.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
