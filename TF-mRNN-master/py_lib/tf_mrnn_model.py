"""Implement the mRNN model with a shared weights strategy in [1, 2].

[1]. Mao, J., Xu, W., Yang, Y., Wang, J., Huang, Z. and Yuille, A., 
Deep captioning with multimodal recurrent neural networks (m-rnn). 
In Proc. ICLR 2015.

[2]. Mao, J., Wei, X., Yang, Y., Wang, J., Huang, Z. and Yuille, A.L., 
Learning like a child: Fast novel visual concept learning from sentence 
descriptions of images. In Proc. ICCV 2015.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging

from common_utils import CommonUtiler

logger = logging.getLogger('TfMrnnModel')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


import tensorflow as tf
from tensorflow.python.ops import math_ops


class mRNNModel(object):
  """The mRNN model with a shared weights strategy in [1, 2]."""

  def __init__(self, is_training, config, num_steps, model_name,
               flag_with_saver=False,
               model_root='./cache/models/mscoco',
               flag_reset_state=False):
    # Set up paths and dirs
    self.cu = CommonUtiler()
    self.model_dir = os.path.join(model_root, model_name)
    self.variable_dir = os.path.join(self.model_dir, 'variables')

    self.cu.create_dir_if_not_exists(self.model_dir)
    self.cu.create_dir_if_not_exists(self.variable_dir)
  
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps
    rnn_size = config.rnn_size
    emb_size = config.emb_size
    vocab_size = config.vocab_size
    vf_size = config.vf_size
    seq_len = config.seq_len
    hidden_img = config.hidden_img

    # Inputs to the model
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    #take 3 d visual features - batch_size,sequence length, input_size
    #self._visual_features = tf.placeholder(tf.float32, [batch_size, vf_size])
    self.input_vf = tf.placeholder(tf.float32, [batch_size, seq_len, vf_size])
    #self.input_vf = tf.placeholder(tf.float32, [batch_size, vf_size])
    self._valid_flags = tf.placeholder(tf.float32, [batch_size, num_steps])
    self._seq_lens = tf.placeholder(tf.int32, [batch_size])

    # Create rnn cell
    if config.rnn_type == 'GRU':
      rnn_cell_basic = tf.nn.rnn_cell.GRUCell(rnn_size)
    elif config.rnn_type == 'LSTM':
      rnn_cell_basic = tf.nn.rnn_cell.LSTMCell(rnn_size, input_size=emb_size, 
          use_peepholes=True)
    else:
      raise NameError("Unknown rnn type %s!" % config.rnn_type)
    if is_training and config.keep_prob_rnn < 1:
      rnn_cell_basic = tf.nn.rnn_cell.DropoutWrapper(
          rnn_cell_basic, output_keep_prob=config.keep_prob_rnn)
    cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_basic] * config.num_rnn_layers)
    state_size = cell.state_size
    
    # Create word embeddings
    self._embedding = embedding = tf.get_variable("embedding", 
        [vocab_size, emb_size])
    inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob_emb < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob_emb)
    
    # Different ways to fuze text and visual information
    if config.multimodal_type == 'mrnn':
      mm_size = config.mm_size
      # Run RNNs
      if flag_reset_state:
        self._initial_state = initial_state = tf.placeholder(tf.float32, 
            [batch_size, state_size])
      else:
        self._initial_state = initial_state = cell.zero_state(
            batch_size, tf.float32)
      inputs = [tf.squeeze(input_, [1])
          for input_ in tf.split(1, num_steps, inputs)]
      outputs_rnn, state = tf.nn.rnn(cell, inputs, 
          initial_state=initial_state,
          sequence_length=self._seq_lens)
      self._final_state = state
      output_rnn = tf.reshape(tf.concat(1, outputs_rnn), [-1, rnn_size])
      
      # Map RNN output to multimodal space
      w_r2m = tf.get_variable("w_r2m", [rnn_size, mm_size])
      b_r2m = tf.get_variable("b_r2m", [mm_size])
      multimodal_l = tf.nn.relu(tf.matmul(output_rnn, w_r2m) + b_r2m)
      
      # Map Visual feature to multimodal space

      #---------------------------------------------------------------
      # LSTM ADDED HERE LSTM ADDED HERE.
      with tf.variable_scope('scope_limited'):
        cell_img = tf.nn.rnn_cell.LSTMCell(hidden_img,state_is_tuple=True)
        val_img, state_img = tf.nn.dynamic_rnn(cell_img, self.input_vf, dtype=tf.float32)
        val_img = tf.transpose(val_img, [1, 0, 2])
        last_img = tf.gather(val_img, int(val_img.get_shape()[0]) - 1)
        self._visual_features = last_img

      #self._visual_features = tf.constant(np.random.rand(batch_size,vf_size), shape=[batch_size, vf_size], dtype="float32")
      #self._visual_features = self.input_vf
      #----------------------------------------------------------------

      w_vf2m = tf.get_variable("w_vf2m", [hidden_img, mm_size])
      b_vf2m = tf.get_variable("b_vf2m", [mm_size])
      mm_vf_single = tf.nn.relu(
          tf.matmul(self._visual_features, w_vf2m) + b_vf2m)
      mm_vf = tf.reshape(tf.tile(mm_vf_single, [1, num_steps]), [-1, mm_size])
      multimodal_l = multimodal_l + mm_vf
      if is_training and config.keep_prob_mm < 1:
        multimodal_l = tf.nn.dropout(multimodal_l, config.keep_prob_mm)
      
      # Map multimodal space to word space
      w_m2w = tf.get_variable("w_m2w", [mm_size, emb_size])
      b_m2w = tf.get_variable("b_m2w", [emb_size])
      output = tf.nn.relu(tf.matmul(multimodal_l, w_m2w) + b_m2w)
      
    elif config.multimodal_type == 'init':
      # Mapping visual feature to the RNN state
      w_vf2state = tf.get_variable("w_vf2state", [vf_size, state_size])
      b_vf2state = tf.get_variable("b_vf2state", [state_size])
      if flag_reset_state:
        self._initial_state = initial_state = tf.placeholder(tf.float32, 
            [batch_size, state_size])
      else:
        self._initial_state = initial_state = tf.nn.relu(
            tf.matmul(self._visual_features, w_vf2state) + b_vf2state)

      # Run RNNs
      inputs = [tf.squeeze(input_, [1])
          for input_ in tf.split(1, num_steps, inputs)]
      outputs_rnn, state = tf.nn.rnn(cell, inputs, 
          initial_state=initial_state,
          sequence_length=self._seq_lens)
      self._final_state = state
      output_rnn = tf.reshape(tf.concat(1, outputs_rnn), [-1, rnn_size])
      
      # Map multimodal space to word space
      w_m2w = tf.get_variable("w_m2w", [rnn_size, emb_size])
      b_m2w = tf.get_variable("b_m2w", [emb_size])
      output = tf.nn.relu(tf.matmul(output_rnn, w_m2w) + b_m2w)
      
    else:
      raise NameError("Unknown multimodal type %s!" % config.multimodal_type)

    # Build sampled softmax loss
    # share the weights between embedding and softmax acc. to [2]
    w_loss = tf.transpose(embedding)
    b_loss = tf.get_variable("b_loss", [vocab_size])
    self._logit = logit = tf.matmul(output, w_loss) + b_loss
    
    target = tf.reshape(math_ops.to_int64(self._targets), [-1])
    valid_flag = tf.reshape(self._valid_flags, [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target)
    self._cost = cost = tf.reduce_sum(loss * valid_flag) / (
        tf.reduce_sum(valid_flag) + 1e-12)
    
    # Create saver if necessary
    if flag_with_saver:
      self.saver = tf.train.Saver(max_to_keep=None)
    else:
      self.saver = None

    # Return the model if it is just for inference
    if not is_training:
      return

    # Create learning rate and gradients optimizer
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    if hasattr(config, 'optimizer'):
      if config.optimizer == 'ori':
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      elif config.optimizer == 'ada': # No GPU
        optimizer = tf.train.AdagradOptimizer(self.lr)
      elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(self.lr)
      elif config.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(self.lr)
      else:
        raise NameError("Unknown optimizer type %s!" % config.optimizer)
    else:
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets
    
  @property
  def valid_flags(self):
    return self._valid_flags

  @property
  def visual_features(self):
    return self._visual_features
    
  @property
  def seq_lens(self):
    return self._seq_lens

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state
    
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
    
  @property
  def embedding(self):
    return self._embedding
    
  @property
  def logit(self):
    return self._logit
