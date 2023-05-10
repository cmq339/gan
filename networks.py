# -*- coding: UTF-8 -*-
from ops import *
from config import *
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
#使用slim库中卷积操作

class generator:       
    '''
    补全网络，
    输入：带mask的具有二进制通道的RGB图像 
    ''' 

    def __call__(self, inputs_img, reuse):
        with tf.variable_scope("G", reuse=reuse) as vs:
            '''
            采用12层卷积网络对原始图片(+mask/去除需要进行填充的部分)进行encoding，得到原图1/16大小的网格
            '''
            inputs = slim.conv2d(inputs_img, 64, 5, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 128, 3, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 128, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = slim.conv2d(inputs, 256, 3, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)        
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)           
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=2)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)          
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=4)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=8)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            inputs = tf.contrib.layers.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity,rate=16)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            

            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)      
            inputs = slim.conv2d(inputs, 256, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)
            
            #对网格采用4层卷积网络进行encoding，得到补全图像
            inputs = slim.conv2d_transpose(inputs, 128, 4, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)  
            inputs = slim.conv2d(inputs, 128, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)            
            inputs = slim.conv2d_transpose(inputs, 64, 4, 2, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs)  
            inputs = slim.conv2d(inputs, 32, 3, 1, activation_fn=tf.identity)
            inputs = slim.batch_norm(inputs, activation_fn=tf.identity)
            inputs = leaky_relu(inputs) 
            out = slim.conv2d(inputs, 3, 3, 1, activation_fn=tf.nn.tanh)
            #输出：RGB图像
        
        G_var = tf.contrib.framework.get_variables('G')
        return out,G_var
        

class discriminator:
    #局部鉴别器和全局鉴别器
    def __call__(self, inputs, inputs_local, train_phase):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('D')]) > 0
      #这句代码是指如果是第一次运行discriminator则reuse=False
        with tf.variable_scope('D',reuse=reuse):
           
            '''
            局部鉴别器
            输入：128x128三通道RGB图片
            4层卷积层和一个全连接层，得到一个1024维向量        
            '''
            local = slim.conv2d(inputs_local, 64, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = slim.conv2d(local, 128, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = slim.conv2d(local, 256, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)            
            local = slim.conv2d(local, 512, 5, 2, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            local = leaky_relu(local)
            local = tf.reshape(local, [-1, np.prod([4, 4, 512])])
            local = slim.fully_connected(local, 1024, activation_fn=tf.identity)
            local = slim.batch_norm(local, activation_fn=tf.identity)
            output_l = leaky_relu(local)


            '''
            全局鉴别器
            输入：256x256三通道RGB图片
            5层卷积层和一个全连接层，得到一个1024维向量
            '''
            image_g = slim.conv2d(inputs, 64, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 128, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 256, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)            
            image_g = slim.conv2d(image_g, 512, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)
            image_g = slim.conv2d(image_g, 512, 5, 2, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            image_g = leaky_relu(image_g)            
            image_g = tf.reshape(image_g, [-1, np.prod([4, 4, 512])])
            image_g = slim.fully_connected(image_g, 1024, activation_fn=tf.identity)
            image_g = slim.batch_norm(image_g, activation_fn=tf.identity)
            output_g = leaky_relu(image_g)
            

            '''
            全连接层
            将全局和局部两个鉴别器输出连接
            '''
            output = tf.concat([output_g,output_l],axis=1)
            output = slim.fully_connected(output, num_outputs=1, activation_fn=None)
            output = tf.squeeze(output, -1) 
            #可以不使用sigmoid函数去激活成(0,1),训练GAN中不使用sigmoid会让网络更稳定
        D_var = tf.contrib.framework.get_variables('D')
        return output,D_var