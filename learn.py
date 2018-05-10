# -*- coding: utf-8 -*-

from game import FEATURE_SIZE, HISTORY_SIZE
from model import DualNetwork
import numpy as np
import tensorflow as tf


def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            grads.append(tf.expand_dims(g, 0))

        grad = tf.reduce_mean(tf.concat(grads, 0), 0)
        v = grad_and_vars[0][1]
        average_grads.append((grad, v))

    return average_grads


def learn(fp_, ckpt_path="", lr_=1e-4, use_gpu=True, gpu_cnt=1):

    device_name = "gpu" if use_gpu else "cpu"
    with tf.get_default_graph().as_default(), tf.device("/cpu:0"):

        # placeholders
        f_list = []
        p_list = []
        r_list = []
        for gpu_idx in range(gpu_cnt):
            f_list.append(tf.placeholder(
                "float", shape=[None, HISTORY_SIZE, FEATURE_SIZE],
                name="feature_%d" % gpu_idx))
            p_list.append(tf.placeholder(
                "float", shape=[None, FEATURE_SIZE], name="prob_%d" % gpu_idx))
            r_list.append(tf.placeholder(
                "float", shape=[None], name="result_%d" % gpu_idx))

        lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        # optimizer and network definition
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        dn = DualNetwork()

        # compute and apply gradients
        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idx in range(gpu_cnt):
                with tf.device("/%s:%d" % (device_name, gpu_idx)):

                    tf.get_variable_scope().reuse_variables()

                    policy_, value_ = dn.model(
                        f_list[gpu_idx], temp=1.0, is_train=True)
                    policy_ = tf.clip_by_value(policy_, 1e-6, 1)

                    loss_p = -tf.reduce_mean(tf.reduce_sum(tf.multiply(
                        p_list[gpu_idx], tf.log(policy_)), 1))
                    loss_v = tf.reduce_mean(
                        tf.square(tf.subtract(value_, r_list[gpu_idx])))
                    if gpu_idx == 0:
                        vars_train = tf.get_collection("vars_train")
                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars_train])
                    loss = loss_p + loss_v + 1e-4 * loss_l2

                    tower_grads.append(opt.compute_gradients(loss))

        train_op = opt.apply_gradients(average_gradients(tower_grads))

        # accuracy
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            with tf.device("/%s:0" % device_name):
                f_acc = tf.placeholder(
                    "float", shape=[None, HISTORY_SIZE, FEATURE_SIZE], name="feature_acc")
                p_acc = tf.placeholder(
                    "float", shape=[None, FEATURE_SIZE], name="prob_acc")
                r_acc = tf.placeholder(
                    "float", shape=[None], name="result_acc")

                p_, v_ = dn.model(f_acc, temp=1.0, is_train=False)
                prediction = tf.equal(tf.argmax(p_, 1), tf.argmax(p_acc, 1))
                accuracy_p = tf.reduce_mean(tf.cast(prediction, "float"))
                accuracy_v = tf.reduce_mean(tf.square(tf.subtract(v_, r_acc)))
                accuracy = (accuracy_p, accuracy_v)

        sess = dn.create_sess(ckpt_path)

    feed = fp_  # FeedPicker
    feed_cnt = feed.size

    # training settings
    batch_cnt = min(100, feed_cnt)
    total_epochs = 4
    epoch_steps = feed_cnt // (batch_cnt * gpu_cnt) + 1
    learning_rate = lr_

    # training
    for epoch_idx in range(total_epochs):
        if epoch_idx > 0:
            learning_rate *= 0.5

        for _ in range(epoch_steps):
            feed_dict_ = {}
            feed_dict_[lr] = learning_rate
            for gpu_idx in range(gpu_cnt):
                batch = feed.next_batch(batch_cnt)

                feed_dict_[f_list[gpu_idx]] = batch[0]
                feed_dict_[p_list[gpu_idx]] = batch[1]
                feed_dict_[r_list[gpu_idx]] = batch[2]

            sess.run(train_op, feed_dict=feed_dict_)

    # calculate accuracy
    acc_batch_cnt = batch_cnt
    acc_steps = feed.size // acc_batch_cnt
    np.random.shuffle(feed._perm)
    acc_sum = [0.0, 0.0]

    str_log = ""
    for _ in range(acc_steps):
        acc_batch = feed.next_batch(acc_batch_cnt)
        accur = sess.run(
            accuracy, feed_dict={f_acc: acc_batch[0],
                                 p_acc: acc_batch[1],
                                 r_acc: acc_batch[2]})
        acc_sum[0] += accur[0]
        acc_sum[1] += accur[1]

    acc_sum[0] *= 100.0 / acc_steps
    acc_sum[1] *= 0.5 / acc_steps

    print("train: accuracy=%3.1f[%%] mse=%.3f"
          % (acc_sum[0], acc_sum[1]))

    str_log += "%3.2f\t%.3f\t" % (acc_sum[0], acc_sum[1])

    # save log
    log_file = open("train_log.txt", "a")
    log_file.write(str_log + "\n")
    log_file.close()

    # save ckpt file
    dn.save_vars(sess, "ckpt/model")
