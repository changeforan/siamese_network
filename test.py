#import system things
import tensorflow as tf
import os

#import helpers
import inference
from inputdata import Player


if __name__ == "__main__":
    players = Player()
    sess = tf.InteractiveSession()
    siamese = inference.Siamese()
    saver = tf.train.Saver()
    model_ckpt = 'model/model.meta'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, 'model/model')
    embed = siamese.o1.eval({siamese.x1: players.test.images})
    embed.tofile('embed.txt')
