# -*- coding: utf-8 -*-

import tensorflow as tf

LAYER_CNT = 1
FEATURE_SIZE = 4
HISTORY_SIZE = 10
INPUT_SIZE = HISTORY_SIZE * FEATURE_SIZE


class DualNetwork(object):

    def get_variable(self, shape_, width_=0.01, name_="weight"):
        var = tf.get_variable(name_, shape=shape_,
                              initializer=tf.random_normal_initializer(
                                  mean=0, stddev=width_))

        if not tf.get_variable_scope()._reuse:
            tf.add_to_collection("vars_train", var)

        return var

    def fully_connected(self, x, output_size, apply_relu=True, batch_norm=True, is_train=False):
        input_size = x.get_shape().as_list()[-1]
        w = self.get_variable([input_size, output_size], name_="weight")

        if batch_norm:
            bn = self.batch_norm(x, is_train)
            connected = tf.matmul(bn, w)
        else:
            b = self.get_variable([output_size], name_="bias")
            connected = tf.matmul(x, w) + b

        return connected if not apply_relu else tf.nn.relu(connected)

    def batch_norm(self, x, is_train, decay=0.99):

        output_size = x.get_shape()[-1]
        scale = tf.get_variable("scale", shape=[output_size],
                                initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", shape=[output_size],
                               initializer=tf.zeros_initializer())
        pop_mean = tf.get_variable("pop_mean", shape=[output_size],
                                   initializer=tf.zeros_initializer(),
                                   trainable=False)
        pop_var = tf.get_variable("pop_var", shape=[output_size],
                                  initializer=tf.ones_initializer(),
                                  trainable=False)

        if not tf.get_variable_scope()._reuse:
            tf.add_to_collection("vars_train", scale)
            tf.add_to_collection("vars_train", beta)
            tf.add_to_collection("vars_train", pop_mean)
            tf.add_to_collection("vars_train", pop_var)

        if is_train:
            batch_mean, batch_var = tf.nn.moments(x, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x,
                                                 batch_mean, batch_var,
                                                 beta, scale, 1e-5)
        else:
            return tf.nn.batch_normalization(x,
                                             pop_mean, pop_var,
                                             beta, scale, 1e-5)

    def model(self, x, temp=1.0, is_train=False):

        h_fc = [tf.reshape(x, [-1, INPUT_SIZE]), ]

        # fully connected layers
        for i in range(LAYER_CNT):
            # [-1, 10 * 4] => [-1, 10 * 4]
            with tf.variable_scope('fc%d' % i):
                h_fc.append(self.fully_connected(
                    h_fc[i], INPUT_SIZE, apply_relu=True,
                    batch_norm=(i != 0), is_train=is_train))

        # policy connection
        with tf.variable_scope('pc'):
            # 1st layer
            # [-1, 10 * 4] => [-1, 4]
            h_pc = self.fully_connected(
                h_fc[LAYER_CNT], FEATURE_SIZE, apply_relu=False,
                batch_norm=True, is_train=is_train)

            # divided by softmax temp and apply softmax
            policy = tf.nn.softmax(tf.div(h_pc, temp), name="policy")

        # value connection
        with tf.variable_scope('vc'):
            # 1st layer
            # [-1, 10 * 4] => [-1, 1]
            h_vc = self.fully_connected(
                h_fc[LAYER_CNT], 1, apply_relu=False,
                batch_norm=True, is_train=is_train)

            # apply hyperbolic tangent
            value = tf.nn.tanh(tf.reshape(h_vc, [-1]), name="value")

        return policy, value

    def create_sess(self, ckpt_path=""):
        # create session
        # read from ckpt files if exists

        with tf.get_default_graph().as_default():

            sess_ = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
            vars_train = tf.get_collection("vars_train")
            v_to_init = list(set(tf.global_variables()) - set(vars_train))

            saver = tf.train.Saver(vars_train, write_version=2)
            if ckpt_path != "":
                saver.restore(sess_, ckpt_path)
                sess_.run(tf.variables_initializer(v_to_init))
            else:
                sess_.run(tf.global_variables_initializer())

        return sess_

    def save_vars(self, sess_, ckpt_path="ckpt/model"):
        # save variables to ckpt files

        with tf.get_default_graph().as_default():

            vars_train = tf.get_collection("vars_train")
            saver = tf.train.Saver(vars_train, write_version=2)
            saver.save(sess_, ckpt_path)
