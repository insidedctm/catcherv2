import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 

def _get_scaled_img(npimg):
  target_size=(432, 368)
  get_base_scale = lambda s, w, h: max(target_size[0] / float(h), target_size[1] / float(w)) * s
  img_h, img_w = npimg.shape[:2]

  # resize
  npimg = cv2.resize(npimg, target_size, interpolation=cv2.INTER_CUBIC)
  return [npimg], [(0.0, 0.0, 1.0, 1.0)]

def get_test_input():
  buffer = np.random.uniform(size=[10,368,432,3])
  img_path = '/Users/robineast/projects/tf-pose-estimation/images/p1.jpg'  
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  print(f"(get_test_input) img.shape={img.shape}")
  print(f"(get_test_input) scaled img shape={_get_scaled_img(img)[0][0].shape}")  
  buffer[0] = _get_scaled_img(img)[0][0]
  return buffer

def load_model():
  input_node = tf.placeholder(tf.float32, shape=(10, 368, 432, 3), name='image')
  y_node = tf.placeholder(tf.int32, shape=[None], name="y")
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
  return training_op, input_node, y_node, logits, xentropy



train_op, input_node, y, logits, output = load_model()
print(output)

init = tf.global_variables_initializer()
# see if we can push an input through the mode
with tf.Session() as sess:
  sess.run(init)
  result1, result_logits = sess.run([output, logits], feed_dict={
                     y: np.ones([10]),
                     input_node: get_test_input()
                   }) 
  print("result: ", result1)
  print("logits: ", result_logits) 
