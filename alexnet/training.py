#coding=utf-8
import os
import numpy as np
import tensorflow as tf

import alexnet.input_data as input_data
import alexnet.model as model

N_CLASSES = 6
IMG_W = 227
IMG_H = 227
BATCH_SIZE = 8
CAPACITY = 200
MAX_STEP = 10000
learning_rate = 0.001
PRINT_STEP = 30
SAVE_STEP = 500

def train(train_dir, logs_train_dir):
  train, train_label = input_data.get_files(train_dir)
  train_batch,train_label_batch = input_data.get_batches_with_onehot(
      train,
      train_label,
      IMG_W,
      IMG_H,
      BATCH_SIZE,
      CAPACITY,
      N_CLASSES
  )
  train_logits = model.inference4train(
          train_batch,
          N_CLASSES
  )
  train_loss = model.losses_with_onehot(train_logits, train_label_batch)
  train_op = model.training(train_loss, learning_rate)
  train_acc = model.evaluation(train_logits, train_label_batch)
  summary_op = tf.summary.merge_all()

  sess = tf.Session()
  train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
      for step in np.arange(MAX_STEP):
          if coord.should_stop():
                  break
          _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
          if step % PRINT_STEP == 0:
              print('Step %d, train loss = %.5f, train accuracy = %.5f%%' %(
                  step, tra_loss, tra_acc*100.0))
              summary_str = sess.run(summary_op)
              train_writer.add_summary(summary_str, step)
          if step % SAVE_STEP == 0 or (step + 1) == MAX_STEP:
              checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)
  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      coord.request_stop()
