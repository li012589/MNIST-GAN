import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from ganLib import GAN

IMG_SIZE = [28,28,1]
Z_SIZE = 100
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0003

SAMPLE_NUM = 6
TF_SAVE_DIR = './TF_SAVE_DIR/'
PIC_SAVE_DIR = './PIC_SAVE/'

def main():
    sess = tf.InteractiveSession()
    gan = GAN(sess,IMG_SIZE,Z_SIZE,G_LEARNING_RATE,D_LEARNING_RATE)
    gan.init()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(TF_SAVE_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        samples = gan.generator(SAMPLE_NUM)
        for i in range(SAMPLE_NUM):
            img = (samples[i,:,:,:].reshape([IMG_SIZE[0],IMG_SIZE[1]]))
            plt.imsave(PIC_SAVE_DIR+'Sampled_'+str(i)+'.png',img)
    else:
        print("Could not find old network weights")

if __name__ == "__main__":
    main()