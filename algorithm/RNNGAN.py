# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime


class RNNGAN(object):
 #change
    def __init__(self,num_batch=64,num_frames=10,n_fft=1024 ,num_rnn_layer = 3, num_hidden_units = [512, 512, 512], tensorboard_directory = 'alogorithm/graphs/RNNGAN', clear_tensorboard = False):

        assert len(num_hidden_units) == num_rnn_layer
        self.num_batch=num_batch
        self.num_frames=num_frames
        self.num_features = n_fft//2+1
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units
        self.EPS = 1e-12
        self.lamda_gan_weight=1.0

        self.gstep = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
        self.gen_learning_rate = tf.placeholder(tf.float32, shape = [], name = 'gen_learning_rate')
        self.dis_learning_rate = tf.placeholder(tf.float32, shape = [], name = 'dis_learning_rate')
        # The shape of x_mixed, y_src1, y_src2 are [batch_size, n_frames (time), n_frequencies]
        self.x_mixed = tf.placeholder(tf.float32, shape = [self.num_batch, self.num_frames, self.num_features], name = 'x_mixed')
        self.y_src2 = tf.placeholder(tf.float32, shape = [self.num_batch, self.num_frames, self.num_features], name = 'y_src2')

        self.y_pred_src2 = self.gen_network_initializer()
        self.dis_fake,self.dis_real=self.dis_network_initializer()

        # Loss balancing parameter
        self.gen_vars = [v for v in tf.trainable_variables() if 'gen_network' in v.name]
        #coding:utf-8 所有生成器的可训练参数
        self.dis_vars = [v for v in tf.trainable_variables() if 'dis_network' in v.name]
        #coding:utf-8 所有判别器的可训练参数
        self.gen_loss = self.gen_loss_initializer()
        self.dis_loss= self.dis_loss_initializer()
        self.gen_optimizer = self.gen_optimizer_initializer()
        self.dis_optimizer = self.dis_optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Tensorboard summary
        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors = True)
            logdir = tensorboard_directory
        else:
            now = datetime.now()
            logdir = os.path.join(tensorboard_directory, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.summary_op = self.summary()


    def gen_network(self):

        rnn_layer = [tf.nn.rnn_cell.GRUCell(size) for size in self.num_hidden_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell, inputs = self.x_mixed, dtype = tf.float32)
        y_hat_src2 = tf.layers.dense(
            inputs = outputs,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y2_hat_src2')
        # Time-frequency masking layer
        # np.finfo(float).eps: the smallest representable positive number such that 1.0 + eps != 1.0
        # Absolute value? In principle y_srcs could only be positive in spectrogram
        # y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        # y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        # Mask with Abs
        y_tilde_src2 = tf.abs(y_hat_src2)

        return y_tilde_src2

    def gen_network_initializer(self):

        with tf.variable_scope('gen_network') as scope:
            y_pred_src2 = self.gen_network()

        return y_pred_src2



    def dis_network(self):
        y_fake=tf.concat([self.x_mixed,self.y_pred_src2],0)
        y_hat1_src2_fake = tf.layers.dense(
                inputs = y_fake,
                units = 1,
                activation = tf.nn.relu,
                name = 'y_hat1_src2')
        y_hat2_src2_fake=tf.reduce_mean(y_hat1_src2_fake,2)
        y_hat_src2_fake=tf.layers.dense(
            inputs=y_hat2_src2_fake,
            units=1,
            activation=tf.nn.sigmoid,
            name='y_hat_src2')
        y_real=tf.concat([self.x_mixed,self.y_src2],0)
        y_hat1_src2_real = tf.layers.dense(
            inputs=y_real,
            units=1,
            activation=tf.nn.relu,
            name='y_hat1_src2',
            reuse=True)
        y_hat2_src2_real = tf.reduce_mean(y_hat1_src2_real, 2)
        y_hat_src2_real = tf.layers.dense(
            inputs=y_hat2_src2_real,
            units=1,
            activation=tf.nn.sigmoid,
            name='y_hat_src2',
            reuse = True)
        return y_hat_src2_fake,y_hat_src2_real

    def dis_network_initializer(self):
        with tf.variable_scope('dis_network') as scope:
            dis_fake,dis_real= self.dis_network()
        return dis_fake, dis_real


    def generalized_kl_divergence(self, y, y_hat):

        return tf.reduce_mean(y * tf.log(y / y_hat) - y + y_hat)

    def gen_loss_initializer(self):
        with tf.variable_scope('gen_loss') as scope:
            # Mean Squared Error Loss
            gen_loss_GAN = tf.reduce_mean(-tf.log(self.dis_fake + self.EPS))  # 计算生成器损失中的GAN_loss部分
            gen_loss_L1 = tf.reduce_mean(tf.abs(self.y_pred_src2-self.y_src2))  # 计算生成器损失中的L1_loss部分
            gen_loss= gen_loss_GAN * self.lamda_gan_weight + gen_loss_L1 * self.lamda_gan_weight  # 计算生成器的loss
        return gen_loss

    def dis_loss_initializer(self):
        with tf.variable_scope('dis_loss') as scope:
            # Mean Squared Error Loss
            dis_loss=tf.reduce_mean(-(tf.log(self.dis_real+self.EPS)+tf.log(1-self.dis_fake+self.EPS)))
        return dis_loss

    def gen_optimizer_initializer(self):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate = self.gen_learning_rate)\
            .minimize(self.gen_loss, global_step = self.gstep,var_list=self.gen_vars)
        return gen_optimizer

    def dis_optimizer_initializer(self):
        dis_optimizer = tf.train.AdamOptimizer(learning_rate = self.dis_learning_rate)\
            .minimize(self.dis_loss, global_step = self.gstep,var_list=self.dis_vars)
        return dis_optimizer

    def train(self, x, y2, gen_learning_rate,dis_learning_rate):
        #step = self.gstep.eval()
        step = self.sess.run(self.gstep)
        _,__, gen_train_loss,dis_train_loss, summaries = self.sess.run([self.gen_optimizer,self.dis_optimizer,
                                                  self.gen_loss, self.dis_loss,self.summary_op],
            feed_dict = {self.x_mixed: x, self.y_src2: y2,
                         self.gen_learning_rate: gen_learning_rate,self.dis_learning_rate:dis_learning_rate})
        self.writer.add_summary(summaries, global_step = step)
        return gen_train_loss,dis_train_loss

    def validate(self, x, y2):

        y2_pred,_,__, validate_gen_loss,validate_dis_loss = self.sess.run([self.y_pred_src2, self.dis_real,self.dis_fake, self.gen_loss,self.dis_loss],
            feed_dict = {self.x_mixed: x, self.y_src2: y2})
        return y2_pred, validate_gen_loss,validate_dis_loss

    def test(self, x):

        y2_pred = self.sess.run([self.y_pred_src2], feed_dict = {self.x_mixed: x})

        return y2_pred

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('dis_loss', self.dis_loss)
            tf.summary.histogram('x_mixed', self.x_mixed)
            tf.summary.histogram('y_src2', self.y_src2)
            summary_op = tf.summary.merge_all()

        return summary_op
