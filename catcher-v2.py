import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 
import aws as aws
import boto3
from data_batch import get_dataflow
import argparse
from NaiveTransferLearningModel import NaiveTransferLearningModel, SummaryWriter

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=200, help='', type=int)
  parser.add_argument('--num_epochs', default=1, help='', type=int)
  parser.add_argument('--restore_path', help='if specified should be a tf.Saver checkpoint file to be loaded')
  return parser.parse_args()

def validate(tf_sess):
  df = get_dataflow("labels_validation.csv", prefix="", shuffle=False)
  all_labels = []
  all_preds = []
  for input, labels in df:
    all_labels.extend(labels) 
    y_pred_logits = tf_sess.run([model.logits], 
                feed_dict= {
                  model.tensor_image: input
                })
    preds = [ 0 if x > y else 1 for (x,y) in y_pred_logits[0]]
    all_preds.extend(preds)

    cond = np.logical_and(labels == 1, np.array(preds) == 0)
    
    false_negative_input = input[cond]
    for op in [img_summ_op, heatmap_summ_op, paf_summ_op]:
      summary_str = sess.run(op, feed_dict={model.tensor_image: false_negative_input})
      file_writer.add_summary(summary_str)
   
  summary_stats = get_summary_stats(all_labels, all_preds)
  
  return summary_stats

def get_summary_stats(labels, preds):
  labels = np.array(labels)
  preds  = np.array(preds)
  return {'validation_accuracy': (len(preds) - np.sum(np.square(preds-labels)))/len(preds)}
  

if __name__ == '__main__':
  args = parse_args()

  # construct and load model, prepare for training
  model = NaiveTransferLearningModel()
  model.load_model(args.restore_path)
  train_op = model.get_training_op()

  # setup summary
  file_writer = SummaryWriter.get_file_writer()

  # validation summary setup
  valid_accuracy_placeholder = tf.placeholder(dtype=tf.float32)
  valid_accuracy = tf.summary.scalar('validation_accuracy', valid_accuracy_placeholder)
 
  # image summary setup
  img_summ_op = tf.summary.image('input_images', model.tensor_image, max_outputs=4)
  heatmap_summ_op = tf.summary.image('hmap_images', tf.expand_dims(model.tensor_output[:,:,:,1], axis=3), max_outputs=4)
  paf_summ_op = tf.summary.image('paf_images', tf.expand_dims(model.tensor_output[:,:,:,20], axis=3), max_outputs=4)
 
  init = tf.global_variables_initializer()
  # see if we can push an input through the mode
  with tf.Session() as sess:
    sess.run(init)
  
    batch_index = 0
    df = get_dataflow("../catcher/labels.csv", batch_size=args.batch_size)
    for epoch in range(args.num_epochs):
 
      for input, labels in df: 
        if batch_index % 100 == 0:
          summary_str = model.loss_summary.eval(feed_dict={model.y_node: labels, model.tensor_image: input}) 
          file_writer.add_summary(summary_str, batch_index)
        _, loss_val = sess.run([train_op, model.loss], feed_dict={model.y_node: labels, model.tensor_image: input})
        batch_index = batch_index + 1
      summary_stats = validate(sess)  
      print(f'accuracy: {summary_stats["validation_accuracy"]:0.2}')

      # write out summary info
      summary_str = valid_accuracy.eval(feed_dict={valid_accuracy_placeholder: summary_stats['validation_accuracy']})
      file_writer.add_summary(summary_str, batch_index)

      # save checkpoint every epoch
      model.save(sess, epoch)

  file_writer.close()
