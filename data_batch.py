import pandas as pd
from tensorpack.dataflow import DataFromGenerator, BatchData, DataFlow
from data import _get_scaled_img, get_frame_nos
from common import get_s3_url
import cv2

class MyDataFlow(DataFlow):
  def __init__(self):
    self.image_cache = {}

  def __iter__(self):
    # load data from somewhere with Python, and yield them
    manifest = pd.read_csv("../catcher/labels.csv")
    # temporarily remove .MOV files from processing
    manifest = manifest[manifest['filename'].apply(lambda x: not x.endswith('.MOV'))]    
    self.prefetch(manifest)
    
    def expand_row(row):
      expand =  list(map(lambda i: (row['filename'], i, 0), range(row['nofall_frame_start'],row['nofall_frame_end']+1)))
      expand_falls =  list(map(lambda i: (row['filename'], i, 1), range(row['fall_frame_start'],row['fall_frame_end']+1)))
      expand.extend(expand_falls)
      return expand

    exploded_manifest = manifest.apply(expand_row, axis=1)
    frame_and_labels = exploded_manifest.explode()

    for item in frame_and_labels:
      cache_key = f"{item[0]}-{item[1]}"
      if cache_key in self.image_cache:
        img = self.image_cache[cache_key]
        yield [img, item[2]]

  def prefetch(self, manifest):
    for ix, item in manifest.iterrows():
      print(f"fetching {item}")
      frame_nos = get_frame_nos(item)
      self.add_images(item['filename'], frame_nos)

  def add_images(self, key, frame_nos, prefix='train/'):
    bucket = 'catcher-videos'
    url = get_s3_url(bucket, f"{prefix}{key}")
    cap = cv2.VideoCapture(url)
    frame_cnt = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      if frame_cnt in frame_nos:
        scaled_frame = _get_scaled_img(frame)[0][0]
        self.image_cache[f"{key}-{frame_cnt}"] = scaled_frame
      frame_cnt = frame_cnt + 1

def get_dataflow(batch_size=50):
  df = MyDataFlow()
  df = BatchData(df, batch_size=batch_size)
  df.reset_state()
  return df

if __name__ == '__main__':

  df = get_dataflow()

  for datapoint in df:
    print(datapoint[0], datapoint[1])
