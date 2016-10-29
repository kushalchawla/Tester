from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import re
import sys
import pickle

sys.path.append('./py_lib/')
from common_utils import CommonUtiler

logger = logging.getLogger('ExpMscoco')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

if __name__ == '__main__':
  # Hyparameters
  min_count = 0
  vocab_path = './cache/dctionary/mscoco_mc%d_vocab' % min_count
  mscoco_root = './datasets/ms_coco'
  #anno_file_names = ['anno_list_mscoco_trainModelVal_m_RNN.npy']
  anno_file_names = ['anno_list_mscoco_trainModelVal_m_RNN.pickle']
  # Preparations
  cu = CommonUtiler()
  cu.create_dir_if_not_exists(os.path.dirname(vocab_path))
  
  # Scan the anno files
  vocab = {}
  for anno_file_name in anno_file_names:
    anno_path = os.path.join(mscoco_root, 'mscoco_anno_files', anno_file_name)
    #annos = np.load(anno_path).tolist()

    input_dict ={}
    with open(anno_path, 'rb') as handle:
      input_dict = pickle.load(handle)
      
    for vf_file_name, annotation in input_dict.items():
      annotation = annotation.strip().split()
      for word in annotation:
        word = word.strip().lower()
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
            
  # Create vocabulary
  with open(vocab_path, 'w') as fout:
    fout.write('<pad>\n')
    fout.write('<unk>\n')
    fout.write('<bos>\n')
    num_word = 3
    for word, freq in vocab.iteritems():
      if freq >= min_count:
        fout.write(word + '\n')
        num_word += 1
  logger.info('%d words in the vocabulary file %s', num_word, vocab_path)
