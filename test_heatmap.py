import tensorflow as tf
from tf_pose.networks import get_network
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.platform import gfile

y_node = tf.placeholder(tf.int32, shape=[None], name="y")
tensor_image = None

def load_model():
  global tensor_image
  GRAPH_PB_PATH = '/Users/robineast/projects/tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb'
  with tf.Session() as sess:
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
  print(tf.get_default_graph().collections)
  for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
    print(ts)
  tensor_image = tf.get_default_graph().get_tensor_by_name('image:0')
  tensor_output = tf.get_default_graph().get_tensor_by_name('Openpose/concat_stage7:0')
  flat1 = tf.reshape(tensor_output, shape=[-1, 46*54*57])
  fc1 = tf.layers.dense(flat1, 64, activation=tf.nn.relu, name="fc1")
  print(tf.get_default_graph().collections)
  logits = tf.layers.dense(fc1, 2, name="output")
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_node)
  loss = tf.reduce_mean(xentropy)
  train_vars = tf.get_default_graph().get_collection('trainable_variables', scope='fc1')
  train_vars.extend(tf.get_default_graph().get_collection('trainable_variables', scope='output'))
  optimizer = tf.train.AdamOptimizer()
  training_op = optimizer.minimize(loss, var_list=train_vars)
  return training_op, y_node, logits, xentropy, tensor_output

_,_,_,_,output = load_model()

img_path = '/Users/robineast/projects/tf-pose-estimation/images/p1.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
width = 432
height = 368
img = cv2.resize(img, (width, height))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output_val = sess.run(output, feed_dict = {tensor_image: [img]})

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
