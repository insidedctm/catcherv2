import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 
import aws as aws
import boto3
from tensorflow.python.platform import gfile
from data_batch import get_dataflow
import argparse
from datetime import datetime

# tensorboard log directory name setup
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=200, help='', type=int)
  parser.add_argument('--num_epochs', default=1, help='', type=int)
  return parser.parse_args()

y_node = tf.placeholder(tf.int32, shape=[None], name="y")
tensor_image = None

def load_model():
  global tensor_image
  GRAPH_PB_PATH = '../tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb'
  with tf.Session() as sess:
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
  tensor_image = tf.get_default_graph().get_tensor_by_name('image:0')
  tensor_output = tf.get_default_graph().get_tensor_by_name('Openpose/concat_stage7:0')
  flat1 = tf.reshape(tensor_output, shape=[-1, 46*54*57])
  fc1 = tf.layers.dense(flat1, 64, activation=tf.nn.relu, name="fc1")
  logits = tf.layers.dense(fc1, 2, name="output")
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_node)
  loss = tf.reduce_mean(xentropy)
  train_vars = tf.get_default_graph().get_collection('trainable_variables', scope='fc1')
  train_vars.extend(tf.get_default_graph().get_collection('trainable_variables', scope='output'))
  optimizer = tf.train.AdamOptimizer()
  training_op = optimizer.minimize(loss, var_list=train_vars)

  loss_summary = tf.summary.scalar('Training_loss', loss)
  file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

  return training_op, tensor_image, y_node, logits, xentropy, tensor_output, loss, loss_summary, file_writer

def validate(tf_sess):
  df = get_dataflow("labels_validation.csv", prefix="", shuffle=False)
  all_labels = []
  all_preds = []
  for input, labels in df:
    all_labels.extend(labels) 
    y_pred_logits = tf_sess.run([logits], 
                feed_dict= {
                  input_node: input
                })
    preds = [ 0 if x > y else 1 for (x,y) in y_pred_logits[0]]
    all_preds.extend(preds)
   
  summary_stats = get_summary_stats(all_labels, all_preds)
  
  return summary_stats

def get_summary_stats(labels, preds):
  labels = np.array(labels)
  preds  = np.array(preds)
  return {'validation_accuracy': (len(preds) - np.sum(np.square(preds-labels)))/len(preds)}
  

if __name__ == '__main__':
  args = parse_args()
  train_op, input_node, y, logits, output, _, loss, loss_summary, file_writer = load_model()
  print(output)

  # validation summary setup
  valid_accuracy_placeholder = tf.placeholder(dtype=tf.float32)
  valid_accuracy = tf.summary.scalar('validation_accuracy', valid_accuracy_placeholder)
  
  init = tf.global_variables_initializer()
  # see if we can push an input through the mode
  with tf.Session() as sess:
    sess.run(init)
  
    batch_index = 0
    df = get_dataflow("../catcher/labels.csv", batch_size=args.batch_size)
    for epoch in range(args.num_epochs):
 
      for input, labels in df: 
        if batch_index % 10 == 0:
          summary_str = loss_summary.eval(feed_dict={y_node: labels, tensor_image: input}) 
          file_writer.add_summary(summary_str, batch_index)
        _, loss_val = sess.run([train_op, loss], feed_dict={y_node: labels, tensor_image: input})
        batch_index = batch_index + 1
      summary_stats = validate(sess)  
      print(f'accuracy: {summary_stats["validation_accuracy"]:0.2}')
      summary_str = valid_accuracy.eval(feed_dict={valid_accuracy_placeholder: summary_stats['validation_accuracy']})
      file_writer.add_summary(summary_str, batch_index)
