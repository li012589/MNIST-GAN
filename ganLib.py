import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weightVariable(name,shape):
    initial = tf.get_variable(name,shape,dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    return initial

def biasVariable(name,shape):
    initial = tf.get_variable(name,shape,initializer=tf.constant_initializer(0.01))
    return initial

def discriminator(image,reuseMark=None):
    with tf.variable_scope("discriminator",reuse = reuseMark):
        wConv1 = weightVariable("wConv1",[5,5,1,32])
        bConv1 = biasVariable("bConv1",[32])

        wConv2 = weightVariable("wConv2",[5,5,32,64])
        bConv2 = biasVariable("bConv2",[64])

        wFC1 = weightVariable("wFC1",[3136,1024])
        bFC1 = biasVariable("bFC1",[1024])

        wFC2 = weightVariable("wFC2",[1024,1])
        bFC2 = biasVariable("bFC2",[1])

        conv1 = tf.nn.conv2d(image, wConv1, strides = [1,1,1,1], padding = 'SAME') + bConv1
        conv1 = tf.nn.avg_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

        conv2 = tf.nn.conv2d(conv1, wConv2, strides = [1,1,1,1], padding = 'SAME') + bConv2
        conv2 = tf.nn.avg_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

        fc1 = tf.reshape(conv2,[-1,3136])
        fc1 = tf.matmul(fc1, wFC1) + bFC1
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc1,wFC2) + bFC2
    return fc2

def generator(z, zDim,reuseMark = None):
    with tf.variable_scope("generator",reuse = reuseMark):
        wFC1 = weightVariable("wFC1",[zDim, 3136])
        bFC1 = biasVariable("bFC1",[3136])

        wConv1 = weightVariable("wConv1",[3,3,1,zDim/2])
        bConv1 = biasVariable("bConv1",[zDim/2])

        wConv2 = weightVariable("wConv2",[3,3,zDim/2,zDim/4])
        bConv2 = biasVariable("bConv2",[zDim/4])

        wConv3 = weightVariable("wConv3",[1,1,zDim/4,1])
        bConv3 = biasVariable("bConv3",[1])

        fc1 = tf.matmul(z,wFC1) + bFC1
        fc1 = tf.reshape(fc1,[-1,56,56,1])
        fc1 = tf.contrib.layers.batch_norm(fc1,epsilon = 1e-5)
        fc1 = tf.nn.relu(fc1)

        conv1 = tf.nn.conv2d(fc1, wConv1, strides = [1,2,2,1], padding = 'SAME') + bConv1
        conv1 = tf.contrib.layers.batch_norm(conv1,epsilon = 1e-5)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.image.resize_images(conv1, [56, 56]) #Unpooling -transposed 2*2 avg_pooling
        #conv1 = tf.reshape(conv1,[-1,56,56,-1])

        conv2 = tf.nn.conv2d(conv1, wConv2, strides = [1,2,2,1], padding = 'SAME') + bConv2
        conv2 = tf.contrib.layers.batch_norm(conv2,epsilon = 1e-5)
        conv2 = tf.nn.relu(conv2)
        #conv2 = tf.reshape(conv2, [-1,56,56])
        conv2 = tf.image.resize_images(conv2, [56, 56])

        conv3 = tf.nn.conv2d(conv2, wConv3, strides = [1,2,2,1], padding = 'SAME') + bConv3
        conv3 = tf.sigmoid(conv3)
    return conv3

class GAN:
    def __init__(self,sess,IMGsize,zSize,gLearningRate,dLearningRate):
        self.sess = sess
        self.zSize = zSize
        self.IMGsize = IMGsize

        self.d = tf.placeholder(tf.float32,[None,IMGsize[0],IMGsize[1],IMGsize[2]])
        self.batchSize = tf.placeholder(tf.int32)
        self.radomZ = tf.random_normal([self.batchSize,zSize],mean = 0,stddev= 1)
        self.D = discriminator(self.d,None)
        self.discriminatorVar = [i for i in tf.trainable_variables() if "discriminator" in i.name]
        self.G = generator(self.radomZ,zSize)
        self.generatorVar = [i for i in tf.trainable_variables() if "generator" in i.name and "BatchNorm" not in i.name]
        self.DG = discriminator(self.G,True)

        self.lossTrue = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D, labels = tf.ones_like(self.D)))
        self.lossFake = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.DG, labels = tf.zeros_like(self.DG)))
        self.lossG = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.DG, labels = tf.ones_like(self.DG)))

        self.dTrueOP = tf.train.AdamOptimizer(dLearningRate).minimize(self.lossTrue, var_list = self.discriminatorVar)
        self.dFakeOP = tf.train.AdamOptimizer(dLearningRate).minimize(self.lossFake, var_list = self.discriminatorVar)
        self.gOP = tf.train.AdamOptimizer(gLearningRate).minimize(self.lossG, var_list = self.generatorVar)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def discriminator(self,data):
        return self.sess.run(self.D,feed_dict={self.d:data})

    def generator(self,batchSize):
        return self.sess.run(self.G,feed_dict={self.batchSize:batchSize})

    def trainD(self,trueData,batchSize):
        self.sess.run(self.dTrueOP,feed_dict={self.d:trueData})
        self.sess.run(self.dFakeOP,feed_dict={self.batchSize:batchSize})

    def trainG(self,batchSize):
        self.sess.run(self.gOP,feed_dict={self.batchSize:batchSize})

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST/")
    image = mnist.train.next_batch(1)[0].reshape([1,28,28,1])
    #print image.shape
    sess = tf.InteractiveSession()
    gan = GAN(sess,[28,28,1],100,0.0001,0.0003)
    gan.init()
    d = gan.discriminator(image)
    #tmp = sess.run(tf.random_normal([1,100],mean = 0,stddev = 1))
    g = gan.generator(1)
    p1 = g[0,:,:,:].reshape([28,28])
    plt.imshow(p1)
    print d
    gan.trainD(image,1)
    gan.trainG(1)
    #images = sess.run()
    plt.show()
    while True:
        pass
