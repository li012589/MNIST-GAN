import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from ganLib import GAN

WRITE_SUMMARY = True
WRITE_PIC = False

IMG_SIZE = [28,28,1]
Z_SIZE = 100
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0003

PRE_TRAIN_D = 3000
TRIAN_TIMES = 1000000
BATCH_SIZE = 50

TF_SAVE_DIR = './TF_SAVE_DIR/'
PIC_SAVE_DIR = './PIC_SAVE/'
SUMMARY_DIR = './summary/'
SAVE_PER_STEP = 20
SUMMARY_PER_STEP = 4
SAMPLE_NUM = 3

def main():
    mnist = input_data.read_data_sets("MNIST/")
    sess = tf.InteractiveSession()
    gan = GAN(sess,IMG_SIZE,Z_SIZE,G_LEARNING_RATE,D_LEARNING_RATE)
    gan.init()
    if WRITE_SUMMARY:
        tf.summary.scalar("G loss",gan.lossG)
        tf.summary.scalar("D loss on real date",gan.lossTrue)
        tf.summary.scalar("D loss on fake date",gan.lossFake)
        sampleIMG = gan.generator(SAMPLE_NUM)
        tf.summary.image("Samples",sampleIMG)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(TF_SAVE_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    t = 0
    state = "pre-train"
    for i in xrange(PRE_TRAIN_D):
        t+=1
        realDate = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]])
        gan.trainD(realDate,BATCH_SIZE)

        print("TimeStep",t,"/ State",state)
        if t % SAVE_PER_STEP == 0:
            saver.save(sess, TF_SAVE_DIR + '-gan', global_step = t)
            print("Saved")

        if WRITE_SUMMARY and t % SUMMARY_PER_STEP == 0:
            summary = sess.run(merged,feed_dict={gan.d:realDate,gan.batchSize:BATCH_SIZE})
            writer.add_summary(summary,t)

    state = 'train'
    for i in xrange(TRIAN_TIMES):
        t+=1
        realDate = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]])
        gan.trainD(realDate,BATCH_SIZE)
        gan.trainG(BATCH_SIZE)
        print("TimeStep",t,"/ State",state)

        if WRITE_SUMMARY and t % SUMMARY_PER_STEP == 0:
            summary = sess.run(merged,feed_dict={gan.d:realDate,gan.batchSize:BATCH_SIZE})
            writer.add_summary(summary,t)

        if t % SAVE_PER_STEP == 0:
            saver.save(sess, TF_SAVE_DIR + '-gan', global_step = t)
            print("Saved")
            if WRITE_PIC and WRITE_SUMMARY:
                for i in range(SAMPLE_NUM):
                    img = (sampleIMG[i,:,:,:].reshape([IMG_SIZE[0],IMG_SIZE[1]]))
                    plt.imsave(PIC_SAVE_DIR+str(t)+'.png',img)

if __name__ == "__main__":
    main()
