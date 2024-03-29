import tensorflow as tf
from tensorflow.python.platform import gfile
from datetime import datetime
from NaiveTransferLearningModel import SummaryWriter

class Conv3DModel:
  def __init__(self):
    self.tensor_image = None
    
    self.saver = None
    self.MODEL_NAME = 'CatcherConv3D'

  def load_model(self, sess, restore_path=None, is_training=True):
      depth=3
      self.input_images = []
      self.openpose_outputs = []
      if restore_path:
        imported_graph_saver = tf.train.import_meta_graph(f'{restore_path}.meta')
        imported_graph_saver.restore(sess, restore_path)
        for ix in range(depth):
          sub_graph_name = f'MyImage{ix}'
          image = tf.get_default_graph().get_tensor_by_name(f'{sub_graph_name}/image:0')
          self.input_images.append(image)
          self.openpose_outputs.append(tf.get_default_graph().get_tensor_by_name(f'{sub_graph_name}/Openpose/concat_stage7:0'))
        self.y_node = tf.get_default_graph().get_tensor_by_name('y:0')
        self.tensor_image = tf.get_default_graph().get_tensor_by_name(f'MyImage0/image:0')
        self.tensor_output = tf.get_default_graph().get_tensor_by_name(f'MyImage0/Openpose/concat_stage7:0')
        self.logits = tf.get_default_graph().get_tensor_by_name('logits/BiasAdd:0') 
        self.loss   = tf.get_default_graph().get_tensor_by_name('loss:0')
        print('model restored')          
      else:
        for ix in range(depth):
          sub_graph_name = f'MyImage{ix}'
          self.create_model_with_frozen_weights(sess, sub_graph_name=sub_graph_name)
          image = tf.get_default_graph().get_tensor_by_name(f'{sub_graph_name}/image:0')
          self.input_images.append(image)
          self.openpose_outputs.append(tf.get_default_graph().get_tensor_by_name(f'{sub_graph_name}/Openpose/concat_stage7:0'))
        self.tensor_image = self.input_images[0]
        self.tensor_output = self.openpose_outputs[0]
        self.y_node = tf.placeholder(tf.int32, shape=[None], name="y")
        hmap_paf_tensor = tf.concat([tf.expand_dims(out, 1) for out in self.openpose_outputs], 1)
        conv3d1 = tf.layers.conv3d(hmap_paf_tensor, 32, 3)
        flat1 = tf.reshape(conv3d1, shape=[-1, 44*52*32])
        fc1 = tf.layers.dense(flat1, 64, activation=tf.nn.relu, name="fc1")
        self.logits = tf.layers.dense(fc1, 2, name="logits")
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_node)
        self.loss = tf.reduce_mean(xentropy, name="loss")
        print('model created')

      if is_training:
        self.train_vars = tf.get_default_graph().get_collection('trainable_variables', scope='fc1')
        self.train_vars.extend(tf.get_default_graph().get_collection('trainable_variables', scope='output'))
        self.loss_summary = self.add_summary_scalar('Training_loss', self.loss)

      self.get_training_op()

      # create Saver
      self.saver = tf.train.Saver()

  def add_summary_scalar(self, summ_name, value):
    try:
      summary_tensor = tf.get_default_graph().get_tensor_by_name(f'{summ_name}:0')
    except:
      print(f'{summ_name} summary tensor not found, creating ...')
      summary_tensor = tf.summary.scalar(summ_name, value)
    return summary_tensor

  def add_summary_image(self, summ_name, value, max_outputs):
    try:
      summary_tensor = tf.get_default_graph().get_tensor_by_name(f'{summ_name}:0')
    except:
      print(f'{summ_name} summary tensor not found, creating ...')
      summary_tensor = tf.summary.image(summ_name, value, max_outputs=max_outputs)
    return summary_tensor

  def create_model_with_frozen_weights(self, sess, sub_graph_name):
    GRAPH_PB_PATH = '../tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb'
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name=sub_graph_name)   
 
  def save(self, sess, epoch, save_path='checkpoint/', experiment_name=None):
    if experiment_name:
      checkpoint_path = f'{save_path}{self.MODEL_NAME}_{experiment_name}_{epoch}.ckpt' 
    else:
      checkpoint_path = f'{save_path}{self.MODEL_NAME}_{epoch}.ckpt'
    print(f'saving checkpoint to {checkpoint_path}, epoch={epoch}, save_path={save_path}, experiment_name={experiment_name}')
    save_path = self.saver.save(sess, checkpoint_path)
    print(f'checkpoint saved to {save_path}')

  def get_training_op(self):
    try:
      self.training_op = tf.get_default_graph().get_operation_by_name('Adam')
    except:
      optimizer = tf.train.AdamOptimizer()
      self.training_op = optimizer.minimize(self.loss, var_list=self.train_vars)
