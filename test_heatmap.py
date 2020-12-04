import tensorflow as tf
from tf_pose.networks import get_network
import cv2
import matplotlib.pyplot as plt
import numpy as np

input_node = tf.placeholder(tf.float32, shape=(None, 368, 432, 3), name='image')
y_node = tf.placeholder(tf.int32, shape=[None], name="y")

def load_model():
  net, pretrain_path, last_layer = get_network('mobilenet_thin', input_node)
  print(tf.get_default_graph().collections)
  flat1 = tf.reshape(net.get_output(), shape=[-1, 46*54*57])
  fc1 = tf.layers.dense(flat1, 64, activation=tf.nn.relu, name="fc1")
  logits = tf.layers.dense(fc1, 2, name="output")
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_node)
  loss = tf.reduce_mean(xentropy)
  train_vars = tf.get_default_graph().get_collection('trainable_variables', scope='fc1')
  train_vars.extend(tf.get_default_graph().get_collection('trainable_variables', scope='output'))
  optimizer = tf.train.AdamOptimizer()
  training_op = optimizer.minimize(loss, var_list=train_vars)
  return training_op, input_node, y_node, logits, xentropy, net.get_output()

_,_,_,_,_,output = load_model()

img_path = '/Users/robineast/projects/tf-pose-estimation/images/p1.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
width = 432
height = 368
img = cv2.resize(img, (width, height))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output_val = sess.run(output, feed_dict = {input_node: [img]})

print(output_val.shape)

fig = plt.figure()
a = fig.add_subplot(2, 2, 2)
tmp = np.amax(output_val[0, :, :, 0:17], axis=2)
print(tmp.shape)
plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)

a = fig.add_subplot(2, 2, 3)
tmp = np.amax(output_val[0, :, :, 19:-1:2], axis=2)
print(tmp.shape)
plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)

a = fig.add_subplot(2, 2, 4)
tmp = np.amax(output_val[0, :, :, 20:-1:2], axis=2)
print(tmp.shape)
plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)

plt.show()
