import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 
import aws as aws
import boto3
from data import get_validation_data

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
  return training_op, input_node, y_node, logits, xentropy

def validate(tf_sess):
  input, labels = get_validation_data()
  y_pred_logits = tf_sess.run([logits], 
              feed_dict= {
                input_node: input
              })
  print('y_pred_logits: ', y_pred_logits)
  preds = [ 0 if x > y else 1 for (x,y) in y_pred_logits[0]]
  print('pred: ', preds)

train_op, input_node, y, logits, output = load_model()
print(output)

init = tf.global_variables_initializer()
# see if we can push an input through the mode
with tf.Session() as sess:
  sess.run(init)
  
  validate(sess)  

