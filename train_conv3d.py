import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tf_pose.networks import get_network
import tensorflow as tf
import numpy as np
import cv2 
import aws as aws
import boto3
from data_batch import get_video_clips_dataflow
import argparse
from Conv3DModel import Conv3DModel, SummaryWriter

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=200, help='', type=int)
  parser.add_argument('--num_epochs', default=1, help='', type=int)
  parser.add_argument('--restore_path', help='if specified should be a tf.Saver checkpoint file to be loaded')
  parser.add_argument('--experiment_label', default=None, help='Identifying text to be added to checkpoint filename')
  return parser.parse_args()

def validate(tf_sess, batch_size=50):
  df = get_video_clips_dataflow("labels_validation.csv", batch_size=50, prefix="", shuffle=False, depth=3)
  all_labels = []
  all_preds = []
  for input, labels in df:
    all_labels.extend(labels) 
    feed_dict = {}
    feed_dict[model.y_node] = labels
    for ix in range(len(model.input_images)):
      input_tensor = model.input_images[ix]
      feed_dict[input_tensor] = input[:,ix,:,:,:]
    y_pred_logits = tf_sess.run([model.logits], 
                feed_dict= feed_dict)
    preds = [ 0 if x > y else 1 for (x,y) in y_pred_logits[0]]
    all_preds.extend(preds)

    cond = np.logical_and(labels == 1, np.array(preds) == 0)
    
    false_negative_input = input[cond]
    for op in [img_summ_op, heatmap_summ_op, paf_summ_op]:
      summary_str = sess.run(op, feed_dict={model.tensor_image: false_negative_input[:,0,:,:,:]})
      file_writer.add_summary(summary_str)
   
  summary_stats = get_summary_stats(all_labels, all_preds)
  
  return summary_stats

def get_summary_stats(labels, preds):
  labels = np.array(labels)
  preds  = np.array(preds)
  TP = np.sum(np.logical_and(labels == 1, preds == 1))
  FN = np.sum(np.logical_and(labels == 1, preds == 0))
  TN = np.sum(np.logical_and(labels == 0, preds == 0))
  FP = np.sum(np.logical_and(labels == 0, preds == 1))
  print(f'TP={TP}; FN={FN}; TN={TN}; FP={FP}')
  accuracy    = (TP+TN)/(TP+TN+FN+FP)
  sensitivity = TP/(TP+FN)
  precision   = TP/(TP+FP)
  specificity = TN/(TN+FP)
  f1_score    = 2. * sensitivity * precision / (sensitivity + precision)

  summary_stats = {
       'validation_accuracy': accuracy,
       'validation_sensitivity': sensitivity,
       'validation_precision': precision,
       'validation_specificity': specificity,
       'validation_f1': f1_score 
  }
  print(f"(get_summary_stats): summary_stats={summary_stats}")
  return summary_stats

if __name__ == '__main__':
  args = parse_args()
  sess = tf.Session()

  # construct and load model, prepare for training
  model = Conv3DModel()
  model.load_model(sess, args.restore_path)

  # setup summary
  file_writer = SummaryWriter.get_file_writer()

  # validation summary setup
  try:
    valid_accuracy_placeholder = tf.get_default_graph().get_tensor_by_name('valid_accuracy_ph:0')
  except:
    valid_accuracy_placeholder = tf.placeholder(dtype=tf.float32, name='valid_accuracy_ph')
  valid_accuracy = model.add_summary_scalar('validation_accuracy', valid_accuracy_placeholder)
 
  # image summary setup
  img_summ_op = model.add_summary_image('input_images', model.tensor_image, max_outputs=4)
  heatmap_summ_op = model.add_summary_image('hmap_images', tf.expand_dims(model.tensor_output[:,:,:,1], axis=3), max_outputs=4)
  paf_summ_op = model.add_summary_image('paf_images', tf.expand_dims(model.tensor_output[:,:,:,20], axis=3), max_outputs=4)
 
  init = tf.global_variables_initializer()
  # see if we can push an input through the mode
  if not args.restore_path:
    print("running initializers")
    sess.run(init)

  batch_index = 0
  df = get_video_clips_dataflow("../catcher/labels.csv", batch_size=args.batch_size, depth=3)
  for epoch in range(args.num_epochs):
 
    for input, labels in df: 

      # construct feed dict and pass into training step
      feed_dict = {}
      feed_dict[model.y_node] = labels
      for ix in range(len(model.input_images)):
        input_tensor = model.input_images[ix]
        feed_dict[input_tensor] = input[:,ix,:,:,:]
      if batch_index % 100 == 0:
        tensor_image = input[:,0,:,:,:]
        summary_str = model.loss_summary.eval(session=sess, feed_dict=feed_dict) 
        file_writer.add_summary(summary_str, batch_index)
      _, loss_val = sess.run([model.training_op, model.loss], feed_dict=feed_dict)
      batch_index = batch_index + 1
    summary_stats = validate(sess, batch_size=args.batch_size)  
    print(f'accuracy: {summary_stats["validation_accuracy"]:0.2}')

    # write out summary info
    summary_str = valid_accuracy.eval(session=sess, feed_dict={valid_accuracy_placeholder: summary_stats['validation_accuracy']})
    file_writer.add_summary(summary_str, batch_index)

    # save checkpoint every epoch
    model.save(sess, epoch, experiment_name=args.experiment_label)

  file_writer.close()
