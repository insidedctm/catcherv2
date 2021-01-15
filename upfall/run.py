import os, sys
parent_dir = "/".join(os.getcwd().split("/")[:-1])
sys.path.append(parent_dir)
print(sys.path)

import boto3
import argparse
from zipfile import ZipFile 
import glob
import cv2 as cv
from joblib import load
from decision_helper import DecisionHelper
import re
import tensorflow as tf
from Conv3DFallClassifier import Conv3DFallClassifier, FifoClipBuffer


BUCKET = 'datasets.xense.co.uk'
KEY_PREFIX = 'UPFall/'
DOWNLOAD_DIR = 'upfall_data'

# The following frames are corrupted and should not be processed
skip_frames = ["upfall_data/2018-07-06T12_03_04.483526.png"]

def main(args):
  print(f"Using args: {args}")
  results_filename = 'Results.csv'
  results_file = open(results_filename, 'w')

  # setup classifier
  threshold=6
  sess = tf.Session()
  clf = Conv3DFallClassifier(sess, "../checkpoint/CatcherConv3D_5.ckpt", threshold)

  keys = list_dataset()
  for key in keys:
    filename = key.split('/')[-1]
    if should_process_file(filename, args):
      print(f'Processing {filename}')
      if not args.listonly:
        download_and_extract(filename)
        result = process_files(sess, clf)
        print("Result is {}".format(result))
        results_file.write(f'{filename},{result}\n')
        remove_files()
  results_file.close()

def should_process_file(filename, args):
  regex = 'Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+).zip'  
  match = re.match(regex, filename)
  subject = int(match.group(1))
  activity = int(match.group(2))
  trial = int(match.group(3))
  camera = int(match.group(4))
  return subject in args.subject and \
         activity in args.activity and \
         trial in args.trial and \
         camera in args.camera

def list_dataset(prefix='UPFall/'):
    files = []
    s3_resource = boto3.resource('s3')
    datasets_bucket = s3_resource.Bucket(name='datasets.xense.co.uk')
    for obj in datasets_bucket.objects.all():
        if obj.key.startswith(prefix):
            files.append(obj.key)
    return files

def process_files(sess, clf, threshold=6):
  # setup buffer to receive frames
  clf.fifo = FifoClipBuffer(buffer_size=3)

  # setup decision helper
  amber_threshold = 2
  threshold = 6
  decision_helper = DecisionHelper(amber_threshold=amber_threshold, red_threshold=threshold)

  for f in sorted(glob.glob("{}/*.png".format(DOWNLOAD_DIR))):
    if f in skip_frames:
      print(f"skipping {f}")
      continue
    frame = cv.imread(f, cv.IMREAD_COLOR)
    prediction = clf.predict(clf.sess, clf.preprocess_frame(frame))
    if prediction:
        decision_helper.apply(1)
    else:
        # No fall detected in this frame - signal this to the decision helper
        decision_helper.apply(0)

    cv.imshow('image',frame)
    cv.waitKey(1)
  cv.destroyAllWindows()
  return decision_helper.is_fall_detected    

def download_and_extract(filename='Subject1Activity2Trial1Camera1.zip'):
  full_key = f'{KEY_PREFIX}{filename}'
  print("Processing {}/{}".format(BUCKET, full_key))

  if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
  s3 = boto3.client('s3')
  download_filename = '{}/{}'.format(DOWNLOAD_DIR, filename)
  s3.download_file(BUCKET, full_key, download_filename)
  with ZipFile(download_filename, 'r') as zip:
    #zip.printdir()
    zip.extractall(path=DOWNLOAD_DIR)
  os.remove(download_filename)  

def remove_files():
  files = glob.glob('{}/*.png'.format(DOWNLOAD_DIR))
  for f in files:
    os.remove(f)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--listonly', action='store_true', help='lists files to be processed (no processing)')
  parser.add_argument('--subject', nargs='+', type=int, help='space-separated list of subjects (1-17), default=all')
  parser.add_argument('--activity', nargs='+', type=int, help='space-separated list of activities (1-11), default=all')
  parser.add_argument('--trial', nargs='+', type=int, help='space-separated list of trials (1-3), default=all')
  parser.add_argument('--camera', nargs='+', type=int, help='space-separated list of cameras (1-2), default=all')

  args = parser.parse_args()
  args.subject = list(range(1,18)) if args.subject is None else args.subject
  args.activity = list(range(1,12)) if args.activity is None else args.activity
  args.trial = list(range(1,4)) if args.trial is None else args.trial
  args.camera = list(range(1,3)) if args.camera is None else args.camera

  return args

if __name__ == '__main__':
  args = parse_args()
  main(args)
