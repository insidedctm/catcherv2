import cv2
import numpy as np
from common import get_s3_url
import pandas as pd

def get_validation_images(data, bucket='catcher-videos'):
  validation_input = []  
  for el in data:
    key = el['key']
    print(key)
    url = get_s3_url(bucket, key)
    frame_nos = get_frame_nos(el)
    print(frame_nos)
    cap = cv2.VideoCapture(url)
    frame_cnt = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      if frame_cnt in frame_nos:
        print(frame_cnt)
        scaled_frame = _get_scaled_img(frame)[0][0]
        scaled_frame = np.expand_dims(scaled_frame, axis=0)
        validation_input.append(scaled_frame)
      frame_cnt = frame_cnt + 1
  validation_input = np.concatenate( validation_input, axis=0 )
  print(validation_input.shape)
  return validation_input

def get_frame_nos(el):
  nofall_frame_start = el['nofall_frame_start']
  nofall_frame_end = el['nofall_frame_end']
  fall_frame_start = el['fall_frame_start']
  fall_frame_end = el['fall_frame_end']
  result = list(range(nofall_frame_start, nofall_frame_end+1))
  result.extend(range(fall_frame_start, fall_frame_end+1))
  return result 


def get_labels(el):
  print('>>>>get_labels: ')
  nofall_frame_start = el['nofall_frame_start']
  nofall_frame_end = el['nofall_frame_end']
  fall_frame_start = el['fall_frame_start']
  fall_frame_end = el['fall_frame_end']
  n_nofall = nofall_frame_end - nofall_frame_start + 1
  n_fall = fall_frame_end - fall_frame_start + 1
  nofalls = np.zeros(n_nofall)
  falls = np.ones(n_fall)
  print(nofalls.shape)
  print(falls.shape)
  labels = np.concatenate([nofalls, falls])
  print(labels.shape)
  return labels

def package_row(row):
  return {
		'key': row['filename'],
		'labels': get_labels(row),
		'nofall_frame_start': row['nofall_frame_start'],
		'nofall_frame_end': row['nofall_frame_end'],
		'fall_frame_start': row['fall_frame_start'],
		'fall_frame_end': row['fall_frame_end']
          }

def read_validation_labels_file():
  df = pd.read_csv('labels_validation.csv')
  print(df)
  print(df.nofall_frame_start) 
  print(df.nofall_frame_end)
  print(df.fall_frame_start)
  print(df.fall_frame_end )
  print(df.iloc[0])
  print('>>>')
  print([package_row(row) for _,row in df.iterrows()])
  print('<<<')
  return [package_row(row) for _,row in df.iterrows()]

flatten = lambda t: [item for sublist in t for item in sublist]

def get_validation_data():
  data_array = read_validation_labels_file()
  labels = [data['labels'] for data in data_array]
  validation_input = get_validation_images(data_array)
  assert(validation_input.shape[0] == np.concatenate(labels).shape[0])
  return validation_input, np.concatenate(labels) 

def _get_scaled_img(npimg):
  target_size=(432, 368)
  get_base_scale = lambda s, w, h: max(target_size[0] / float(h), target_size[1] / float(w)) * s
  img_h, img_w = npimg.shape[:2]

  # resize
  npimg = cv2.resize(npimg, target_size, interpolation=cv2.INTER_CUBIC)
  return [npimg], [(0.0, 0.0, 1.0, 1.0)]

if __name__ == '__main__':
  print("RESULT:")
  print(get_validation_data())
