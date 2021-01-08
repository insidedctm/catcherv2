from Conv3DModel import Conv3DModel
from decision_helper import DecisionHelper
import numpy as np
import tensorflow as tf
from data import _get_scaled_img
import time
import cv2

class Conv3DFallClassifier:
  def __init__(self, sess, model_path, threshold):
    print("Conv3DFallClassifier::__init__")
    self.model = Conv3DModel()
    self.model.load_model(sess, model_path, is_training=False)
    self.sess = sess
    self.threshold = threshold

  def predict(self, sess, frame):
    fifo = self.fifo
    fifo.add(frame)
    if fifo.len() < fifo.buffer_size:
      # not enough frames in the buffer to run prediction
      return False

    model = self.model
    feed_dict = {}
    start_ix = min(fifo.buffer.keys())
    for ix in range(len(model.input_images)):
      input_tensor = model.input_images[ix]
      feed_dict[input_tensor] = fifo.buffer[start_ix + ix]
    y_pred_logits = sess.run([model.logits],
                feed_dict= feed_dict)
    return False if y_pred_logits[0][0][0] > y_pred_logits[0][0][1] else True


  def decision(self, video, threshold = 6, amber_threshold = 2, visualise_output=False, skip_frames=0):
      '''
      Return a fall/nofall decision for entire video sequence. If visualise_output is set to True this will
      also display a screen showing each detection and informative text. Depending on the setting of
      display_show_frame the function may also show the original video frame
      Args:
          video: A sequence of frames, either a video or a stream from a webcam, cv2.VideoCapture
          threshold: How many frames of fall detection required to trigger fall detection, int [6]
          amber_threshold: How many frames of fall detection required to trigger 'amber' alert, int [2]
          visualise_output: Whether to display output, bool [False]
          skip_frames: not used
      Returns: True if a fall was detected in the video sequence, bool
      '''
      
      # setup new buffer to hold video frames
      self.fifo = FifoClipBuffer(buffer_size=3)

      self.decision_helper = DecisionHelper(amber_threshold=amber_threshold, red_threshold=self.threshold)
      fall_detected = False

      while True:
          prediction = [0]
          ok, frame = video.read()
          if not ok:
              break

          prediction = self.predict(self.sess, self.preprocess_frame(frame))
          if prediction:
              self.decision_helper.apply(1)
          else:
              # No fall detected in this frame - signal this to the decision helper
              self.decision_helper.apply(0)

      return self.decision_helper.is_fall_detected

  def preprocess_frame(self, frame):
    scaled_img = _get_scaled_img(frame)[0][0]
    return np.expand_dims(scaled_img, axis=0)

class FifoClipBuffer:
  def __init__(self, buffer_size):
    """A maximum of buffer_size items will be stored in the buffer. When an item is added to a buffer
    that already has buffer_size items in it the earliest item is dropped
    """
    self.buffer_size = buffer_size
    self.num_items = 0
    self.buffer = {}

  def len(self):
    return len(self.buffer)

  def add(self, frame):
    num_items = self.num_items
    self.buffer[num_items] = frame
    self.num_items = num_items + 1
    if self.len() > self.buffer_size:
      self.remove(self.num_items - self.buffer_size - 1)
      assert self.len() == self.buffer_size, f"after popping item from FifoClipBuffer the size should be {self.buffer_size}"

  def remove(self, ix):
    del self.buffer[ix]

if __name__ == '__main__':
  with tf.Session() as sess:
    clf = Conv3DFallClassifier(sess, "checkpoint/CatcherConv3D_Test3_3.ckpt")
    clf.fifo = FifoClipBuffer(buffer_size=3)
    raw_img = np.zeros((438,536,3))
    tic = time.perf_counter()
    for i in range(10):
      scaled_img = _get_scaled_img(raw_img)[0][0]
      scaled_img = np.expand_dims(scaled_img, axis=0)
      clf.predict(sess, scaled_img)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    print('------------------------------------')

    # test decision method
    for actual in ['fall', 'nofall']:
      video = cv2.VideoCapture(f'/Users/robineast/Downloads/{actual}/p12_1.mp4')
      print(f'Decision for {actual} video: {clf.decision(video)}')
