import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 
import aws as aws
import boto3
from data import get_validation_data
from tensorflow.python.platform import gfile
from data_batch import get_dataflow
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=200, help='')
  parser.add_argument('--num_epochs', default=1, help='')
  return parser.parse_args()

def main(src_bucket, src_key, output_csv):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=src_bucket)
    urls = []
    file_names = []
    for obj in bucket.objects.filter(Prefix=src_key, Delimiter='/'):
        if obj.key == 'train/':
            print('skipping {}'.format(obj.key))
            continue
        urls.append(aws.get_s3_url(src_bucket, obj.key))
        file_names.append(obj.key)

    process(urls, file_names)

def process(files_or_urls, file_names):
    features = []
    human_ids = []
    frame_ids = []
    video_names = []
    for video, video_filename in zip(files_or_urls, file_names):
        print('processing {} (file_name={})'.format(video, video_filename))
        cap = cv2.VideoCapture(video)
        frame_num = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            humans_parts = extract_features_from_frame(frame, e)

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
  print(tf.get_default_graph().collections)
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
  return training_op, tensor_image, y_node, logits, xentropy, tensor_output, loss

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
   
  print_summary(all_labels, all_preds)

def print_summary(labels, preds):
  labels = np.array(labels)
  preds  = np.array(preds)
  print('pred: ', preds)
  print('labels:', labels)
  print('errors: ', preds - labels)
  print('sqerrors: ', np.square(preds-labels))
  print('sse: ', np.sum(np.square(preds-labels)))
  print(f'error rate: {np.sum(np.square(preds-labels))/len(preds):0.2}')
  print(f'accuracy: {(len(preds) - np.sum(np.square(preds-labels)))/len(preds):0.2}')


if __name__ == '__main__':
  args = parse_args()
  train_op, input_node, y, logits, output, _, loss = load_model()
  print(output)
  
  init = tf.global_variables_initializer()
  # see if we can push an input through the mode
  with tf.Session() as sess:
    sess.run(init)
  

    for epoch in range(args.num_epochs):
      df = get_dataflow("../catcher/labels.csv", batch_size=args.batch_size)
  
      for input, labels in df: 
        _, loss_val = sess.run([train_op, loss], feed_dict={y_node: labels, tensor_image: input})
        print(loss_val)
      validate(sess)  
  
