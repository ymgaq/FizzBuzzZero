# -*- coding: utf-8 -*-

from math import sqrt

from game import FizzBuzzGame, num2str, FEATURE_SIZE, HISTORY_SIZE, DRAW_NB
from model import DualNetwork
import numpy as np
import tensorflow as tf


NODE_NB = 2 ** 12  # 4096
EXPAND_THRESHOLD = 1
CANDIDATE_LIST = [-1] + [DRAW_NB + 1 + i for i in range(FEATURE_SIZE - 1)]


class Node(object):
    # class of node of search tree

    def __init__(self):

        # initialize branch information
        # each member has dimension of FEATURE_SIZE
        self.cand = np.array(CANDIDATE_LIST)         # candidates
        self.prob = np.full(FEATURE_SIZE, 0.0)       # probability
        self.value = np.full(FEATURE_SIZE, 0.0)      # raw value after answer
        self.value_win = np.full(FEATURE_SIZE, 0.0)  # sum of value backed up
        self.visit_cnt = np.full(FEATURE_SIZE, 0)    # sum of visit count
        # next node id after answer
        self.next_id = np.full(FEATURE_SIZE, -1)
        # next node hash
        self.next_hash = np.full(FEATURE_SIZE, -1, dtype=np.int64)
        self.evaluated = np.full(FEATURE_SIZE, False)

        # initialize node information
        self.clear()

    def clear(self):

        self.cand[0] = -1
        self.total_value = 0.0  # sum of value backed up during the search
        self.total_visit = 0    # sum of visit count of all branches
        self.hash = 0           # hash of game

    def init_branch(self):
        # not need to be called in clear() because called in create_node()

        self.prob.fill(0.0)
        self.value.fill(0.0)
        self.value_win.fill(0.0)
        self.visit_cnt.fill(0)
        self.next_id.fill(-1)
        self.next_hash.fill(-1)
        self.evaluated.fill(False)


class Tree(object):
    # class of search tree

    cp = 1.0
    stop = False

    def __init__(self, ckpt_path="model.ckpt", use_gpu=True, gpu_idx=0, reuse=False):

        self.set_sess(ckpt_path, use_gpu, gpu_idx, reuse)
        self.node = [Node() for _ in range(NODE_NB)]
        self.clear()

    def clear(self):
        # initialize search tree

        for nd in self.node:
            nd.clear()
        self.node_cnt = 0
        self.root_id = 0
        self.root_num = 0
        self.node_hashs = {}
        self.eval_cnt = 0
        Tree.stop = False

    def set_sess(self, ckpt_path, use_gpu=True, device_idx=0, reuse=False):
        # create the session on the device

        device_name = "gpu" if use_gpu else "cpu"
        with tf.get_default_graph().as_default(), tf.device("/%s:%d" % (device_name, device_idx)):
            dn = DualNetwork()
            if reuse:
                tf.get_variable_scope().reuse_variables()

            self.x = tf.placeholder(
                "float", shape=[None, HISTORY_SIZE, FEATURE_SIZE])
            self.pv = dn.model(self.x, temp=1.0, is_train=False)
            self.sess = dn.create_sess(ckpt_path)

    def evaluate(self, fbg):
        # evaluate policy and value of the input state.

        f_ = np.reshape(fbg.feature(), (1, HISTORY_SIZE, FEATURE_SIZE))
        return self.sess.run(self.pv, feed_dict={self.x: f_})

    def delete_node(self):
        # delete old nodes

        if self.node_cnt < NODE_NB * 0.5:
            return
        for i in range(NODE_NB):
            num = self.node[i].cand[0]
            if 0 <= num and num < self.root_num:
                if self.node[i].hash in self.node_hashs:
                    self.node_hashs.pop(self.node[i].hash)
                self.node[i].clear()

    def create_node(self, fbg, prob):
        # create node

        hs = fbg.hash()

        if hs in self.node_hashs and \
                self.node[self.node_hashs[hs]].hash == hs and \
                self.node[self.node_hashs[hs]].cand[0] == fbg.next_num:

            return self.node_hashs[hs]

        node_id = hs % NODE_NB
        while self.node[node_id].cand[0] != -1:
            node_id = 0 if node_id + 1 >= NODE_NB else node_id + 1
        self.node_hashs[hs] = node_id
        self.node_cnt += 1

        nd = self.node[node_id]
        nd.clear()
        nd.init_branch()

        nd.hash = hs
        nd.cand[0] = fbg.next_num

        for i in range(FEATURE_SIZE):
            nd.prob[i] = prob[i]

        return node_id

    def dirichlet_prob(self, prob_):
        # add dirichled noise to the root node probability

        noise = np.random.dirichlet([0.1 for _ in range(FEATURE_SIZE)])
        return 0.75 * prob_ + 0.25 * noise

    def search_branch(self, fbg, node_id):
        # select and evaluate branch

        nd = self.node[node_id]
        nd_rate = 0.0 if nd.total_visit == 0 else nd.total_value / nd.total_visit

        # calculate action values of all branches at once
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = nd.value_win / nd.visit_cnt  # including dividing by 0
            rate[~np.isfinite(rate)] = nd_rate  # convert nan, inf to nd_rate
        bonus = Tree.cp * nd.prob * sqrt(nd.total_visit) / (nd.visit_cnt + 1)
        action_value = rate + bonus
        best = np.argmax(action_value)

        next_id = nd.next_id[best]  # -1 if not expanded

        # advance the game
        continue_game = fbg.play(nd.cand[best])

        # whether nd is leaf node or not
        leaf_node = not self.has_next(node_id, best)
        leaf_node |= nd.visit_cnt[best] < EXPAND_THRESHOLD
        leaf_node |= not continue_game

        if leaf_node:
            if nd.evaluated[best]:
                value = nd.value[best]
            else:
                prob_, value_ = self.evaluate(fbg)
                self.eval_cnt += 1
                # flip value because it is opponent's value
                value = -value_[0]
                nd.value[best] = value
                nd.evaluated[best] = True

                if continue_game:
                    if self.node_cnt > 0.85 * NODE_NB:
                        self.delete_node()

                    # expand node
                    next_id = self.create_node(fbg, prob_[0])
                    next_nd = self.node[next_id]
                    nd.next_id[best] = next_id
                    nd.next_hash[best] = fbg.hash()

                    # copy value_win and visit_cnt
                    next_nd.total_value -= nd.value_win[best] + value
                    next_nd.total_visit += nd.visit_cnt[best] + 1

        else:
            value = -self.search_branch(fbg, next_id)

        # backup
        nd.total_value += value
        nd.total_visit += 1
        nd.value_win[best] += value
        nd.visit_cnt[best] += 1

        return value

    def search(self, fbg, search_limit, use_dirichlet=True, show_info=False):
        # search for search_limit times

        prob, _ = self.evaluate(fbg)
        self.root_id = self.create_node(fbg, prob[0])
        self.root_num = fbg.next_num

        nd = self.node[self.root_id]
        self.delete_node()
        self.eval_cnt = 0

        # add dirichlet noise to the root node
        prob_org = np.copy(nd.prob)
        if use_dirichlet:
            nd.prob = self.dirichlet_prob(nd.prob)

        # search
        if not self.stand_out(nd):
            fbg_cpy = FizzBuzzGame()
            for i in range(search_limit):
                fbg.copy_to(fbg_cpy)
                self.search_branch(fbg_cpy, self.root_id)

                if self.stand_out(nd):
                    break

        best = np.argmax(nd.visit_cnt)

        # recover prob
        nd.prob = prob_org

        if show_info:
            # show thinking information

            print("\nplay count=%d: evaluated=%d" %
                  (nd.cand[0], self.eval_cnt))
            self.print_info(self.root_id)

        return nd.cand[best]

    def has_next(self, node_id, br_id):
        # return whether node_id/br_id has link to next node

        nd = self.node[node_id]
        next_id = nd.next_id[br_id]
        if next_id < 0:
            return False
        nnd = self.node[next_id]

        return nnd.hash == nd.next_hash[br_id] and nnd.cand[0] == nd.cand[0] + 1

    def stand_out(self, nd):
        # return whether the best branch is standing out

        best, second = tuple((np.argsort(nd.visit_cnt)[::-1])[:2].tolist())
        return nd.total_visit > 100 and nd.visit_cnt[best] > nd.visit_cnt[second] * 20

    def branch_rate(self, nd, id_):
        # return winning rate of id_'s answer
        # ranging within 0.0-1.0

        return nd.value_win[id_] / max(nd.visit_cnt[id_], 1) / 2 + 0.5

    def best_sequence(self, node_id, first_num):
        # return the best sequence from first_num

        seq_str = "%s" % num2str(first_num)

        for _ in range(5):
            nd = self.node[node_id]

            best = np.argmax(nd.visit_cnt)
            if nd.visit_cnt[best] == 0:
                break
            seq_str += "->%s" % num2str(nd.cand[best])

            if not self.has_next(node_id, best):
                break
            node_id = nd.next_id[best]

        return seq_str

    def print_info(self, node_id):
        # print search information

        nd = self.node[node_id]
        ordered_idx = np.argsort(nd.visit_cnt)[::-1]
        print("|candidate|count|rate |value|prob | best sequence")

        for i in range(FEATURE_SIZE):
            m = ordered_idx[i]
            visit_cnt = nd.visit_cnt[m]
            if visit_cnt == 0:
                break

            rate = 0.0 if visit_cnt == 0 else self.branch_rate(nd, m) * 100
            value = (nd.value[m] / 2 + 0.5) * 100

            print("|%-9s|%5d|%5.1f|%5.1f|%5.1f| %s" % (
                num2str(nd.cand[m]), visit_cnt, rate, value, nd.prob[m] * 100,
                self.best_sequence(nd.next_id[m], nd.cand[m])))
