import tensorflow as tf
from tensorflow.python.platform import gfile
from datetime import datetime

class SummaryWriter:
  @staticmethod
  def get_file_writer():
    # tensorboard log directory name setup
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
  
    return file_writer

class NaiveTransferLearningModel:
  def __init__(self):
    print("NaiveTransferLearningModel constructor")
    self.tensor_image = None
    self.y_node = tf.placeholder(tf.int32, shape=[None], name="y")
    self.saver = None
    self.MODEL_NAME = 'NaiveTransferLearner'

  def load_model(self, restore_path=None):
    GRAPH_PB_PATH = '../tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb'
    with tf.Session() as sess:
      with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
      self.tensor_image = tf.get_default_graph().get_tensor_by_name('image:0')
      self.tensor_output = tf.get_default_graph().get_tensor_by_name('Openpose/concat_stage7:0')
      flat1 = tf.reshape(self.tensor_output, shape=[-1, 46*54*57])
      fc1 = tf.layers.dense(flat1, 64, activation=tf.nn.relu, name="fc1")
      self.logits = tf.layers.dense(fc1, 2, name="output")
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_node)
      self.loss = tf.reduce_mean(xentropy)

      self.train_vars = tf.get_default_graph().get_collection('trainable_variables', scope='fc1')
      self.train_vars.extend(tf.get_default_graph().get_collection('trainable_variables', scope='output'))
      self.loss_summary = tf.summary.scalar('Training_loss', self.loss)

      # create Saver
      self.saver = tf.train.Saver()

      if restore_path:
        # restore weights from checkpoint
        self.saver.restore(sess, restore_path)
        print(f'weights restored from {restore_path}')
 
  def save(self, sess, epoch, save_path='checkpoint/'):
    save_path = self.saver.save(sess, f'{save_path}{self.MODEL_NAME}_{epoch}.ckpt')
    print(f'checkpoint saved to {save_path}')

  def get_training_op(self):
    self.optimizer = tf.train.AdamOptimizer()
    self.training_op = self.optimizer.minimize(self.loss, var_list=self.train_vars)
    return self.training_op
