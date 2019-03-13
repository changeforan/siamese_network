#import system things
import tensorflow as tf
import os
import numpy as np
#import helpers
import inference
from inputdata import Player
import visualize

if __name__ == "__main__":
    players = Player()
    sess = tf.InteractiveSession()
    siamese = inference.Siamese()
    saver = tf.train.Saver()
    model_ckpt = 'model/model.meta'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, 'model/model')
    x = players.test.next_batch(50)[0]
    embed = siamese.o1.eval({siamese.x1: x})
    # embed = embed.reshape([-1, 2])
    x = np.array(x).reshape([-1, 128, 128, 3]) * 255
    x = x.astype(int)
    visualize.visualize(embed, x)
