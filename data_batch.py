import pandas as pd
import numpy as np
from tensorpack.dataflow import DataFromGenerator, BatchData, DataFlow
from data import _get_scaled_img, get_frame_nos
from common import get_s3_url
import cv2
import boto3
from os.path import isfile

class MyDataFlow(DataFlow):
  def __init__(self, labels_path="../catcher/labels.csv", prefix="train/", shuffle=True):
    self.image_cache = {}
    self.labels_path = labels_path
    self.prefix = prefix
    self.shuffle = shuffle

  def __iter__(self):
    # load data from somewhere with Python, and yield them
    manifest = pd.read_csv(self.labels_path)
    self.prefetch(manifest)
    
    def expand_row(row):
      expand =  list(map(lambda i: (row['filename'], i, 0), range(row['nofall_frame_start'],row['nofall_frame_end']+1)))
      expand_falls =  list(map(lambda i: (row['filename'], i, 1), range(row['fall_frame_start'],row['fall_frame_end']+1)))
      expand.extend(expand_falls)
      return expand

    exploded_manifest = manifest.apply(expand_row, axis=1)
    frame_and_labels = exploded_manifest.explode()
    if self.shuffle:
      frame_and_labels = frame_and_labels.sample(frac=1)

    for item in frame_and_labels:
      cache_key = f"{item[0]}-{item[1]}"
      if cache_key in self.image_cache:
        img = self.image_cache[cache_key]
        yield [img, item[2]]

  def prefetch(self, df):
    """Iterates through the rows of a pandas data frame. Input data frame contains the columns 'filename', 
    'nofall_frame_start', 'nofall_frame_end', 'fall_frame_start' and 'fall_frame_end'.

    For each row extract the images in the ranges given by the (no)fall_frame_start/end fields and 
    add to the image cache. The key used for each entry in the cache is {key}-{frame number}.
    Arguments:
        df : Input pandas DataFrame.
    Returns:
        Nothing, adds images to image_cache
    """
    for ix, item in df.iterrows():
      frame_nos = get_frame_nos(item)
      self.add_images(item['filename'], frame_nos)

  def add_images(self, key, frame_nos):
    bucket = 'catcher-videos'
    target_dir = 'raw/'
    #url = get_s3_url(bucket, f"{self.prefix}{key}")
    filename = self.get_download_path(target_dir, key)
    self.check_and_download_s3_file(bucket, f"{self.prefix}{key}", filename)
    cap = cv2.VideoCapture(filename)
    frame_cnt = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      if frame_cnt in frame_nos:
        scaled_frame = _get_scaled_img(frame)[0][0]
        self.image_cache[f"{key}-{frame_cnt}"] = scaled_frame
      frame_cnt = frame_cnt + 1

  def check_and_download_s3_file(self, bucket, key, filename):
    if isfile(filename):
      return
    print(f'{filename} is not cached, download ...')
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, filename)
    
  def get_download_path(self, directory, filename):
    if directory.endswith('/'):
      return f'{directory}{filename}'
    else:
      return f'{directory}/{filename}'

class VideoClipsDataFlow(MyDataFlow):
  def __init__(self, labels_path, prefix, shuffle):
    super().__init__()
    
  def __iter__(self):
    # load data from somewhere with Python, and yield them
    manifest = pd.read_csv(self.labels_path)
    self.prefetch(manifest)

    def expand_row(row):
      expand =  list(map(lambda i: (row['filename'], i, 0), range(row['nofall_frame_start'],row['nofall_frame_end']+1)))
      expand_falls =  list(map(lambda i: (row['filename'], i, 1), range(row['fall_frame_start'],row['fall_frame_end']+1)))
      expand.extend(expand_falls)
      return expand

    exploded_manifest = manifest.apply(expand_row, axis=1)
    frame_and_labels = exploded_manifest.explode()
    if self.shuffle:
      frame_and_labels = frame_and_labels.sample(frac=1)

    for item in frame_and_labels:
      cache_key = f"{item[0]}-{item[1]}"
      cache_key_minus1 = f"{item[0]}-{item[1]-1}" 
      cache_key_minus2 = f"{item[0]}-{item[1]-2}"
      if cache_key in self.image_cache:
        if cache_key_minus1 in self.image_cache:
          if cache_key_minus2 in self.image_cache:
            img = self.image_cache[cache_key]
            img_minus1 = self.image_cache[cache_key_minus1]
            img_minus2 = self.image_cache[cache_key_minus2]
            img_concatenated = np.vstack([img[np.newaxis], img_minus1[np.newaxis], img_minus2[np.newaxis]]) 
            yield [img_concatenated, item[2]]

def get_dataflow(labels_path, batch_size=50, prefix="train/", shuffle=True):
  df = MyDataFlow(labels_path, prefix, shuffle)
  df = BatchData(df, batch_size=batch_size, remainder=True)
  df.reset_state()
  return df

def get_video_clips_dataflow(labels_path, batch_size=50, prefix="train/", shuffle=True):
  df = VideoClipsDataFlow(labels_path, prefix, shuffle)
  df = BatchData(df, batch_size=batch_size, remainder=True)
  df.reset_state()
  return df

if __name__ == '__main__':

  train = get_video_clips_dataflow("../catcher/labels.csv")

  for datapoint in train:
    print(datapoint[0], datapoint[1])

  valid = get_video_clips_dataflow("labels_validation.csv", prefix="", shuffle=False)

  for datapoint in valid:
    print(datapoint[0], datapoint[1])


