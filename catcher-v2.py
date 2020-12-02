import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np

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
                     input_node: np.random.uniform(size=[10,368,432,3])
                   }) 
  print("result: ", result1)
  print("logits: ", result_logits) 
