# -*- coding: utf-8 -*-

from sys import stdout

from game import FizzBuzzGame, FEATURE_SIZE, PLAYER_NB, DRAW_NB
import numpy as np
import search


class Feed(object):
    # class of feed for training

    def __init__(self):

        self._feature = []
        self._prob = []
        self._result = []
        self.size = 0
        self.forget_size = [0 for _ in range(10)]

    def append(self, f_, p_, r_):
        # add feature, prob and result

        self._feature.append(f_)
        self._prob.append(p_)
        self._result.append(r_)
        self.size += 1
        self.forget_size[0] += 1

    def forget(self):
        # delete the oldest record

        fs = self.forget_size[-1]
        self._feature = self._feature[fs:]
        self._prob = self._prob[fs:]
        self._result = self._result[fs:]

        self.forget_size.pop()
        self.forget_size.insert(0, 0)
        self.size -= fs

    def get(self):
        # returns numpy array

        return (np.stack(self._feature).astype(np.float32),
                np.stack(self._prob).astype(np.float32),
                np.stack(self._result).astype(np.float32))


class FeedPicker(object):
    # class of random picker for feed

    def __init__(self, feed_):

        self._feature, self._prob, self._result = feed_.get()
        self.size = self._feature.shape[0]
        self._idx = 0
        self._perm = np.arange(self.size)
        np.random.shuffle(self._perm)

    def next_batch(self, batch_size=100):
        if self._idx + batch_size > self.size:
            np.random.shuffle(self._perm)
            self._idx = 0
        start = self._idx
        self._idx += batch_size
        end = self._idx

        # slice for mini-batch
        f_batch = self._feature[self._perm[start:end]]
        p_batch = self._prob[self._perm[start:end]]
        r_batch = self._result[self._perm[start:end]]

        return f_batch, p_batch, r_batch


def feed_match(feed, match_cnt, search_limit, ckpt_path,
               initial_life=1, use_gpu=True, gpu_idx=0,
               reuse=False, show_info=True):

    # delete old feed
    feed.forget()

    tree = search.Tree(ckpt_path, use_gpu, gpu_idx, reuse)
    fbg = FizzBuzzGame(initial_life)
    prob_leaf = np.full((FEATURE_SIZE), 0.0)

    correct_cnt = 0
    play_cnt = 0
    lengths = []

    print("")

    for i in range(match_cnt):

        fbg.clear()
        tree.clear()

        continue_game = True

        while(continue_game and fbg.next_num <= DRAW_NB):

            # show log only in the first match
            show_info = show_info and i == 0
            num = tree.search(fbg, search_limit, True, show_info)

            prob = tree.node[tree.root_id].visit_cnt
            prob = prob if prob.sum() == 0 else prob.astype(float) / prob.sum()
            feed.append(fbg.feature(), prob, 0)

            if fbg.legal(num, fbg.next_num):
                correct_cnt += 1
            play_cnt += 1

            continue_game = fbg.play(num)

        stdout.write("\r%03d/%03d games" % (i + 1, match_cnt))
        stdout.flush()

        is_draw = np.count_nonzero(fbg.lives) == PLAYER_NB
        continue_length = DRAW_NB if is_draw else fbg.next_num - 2
        lengths.append(continue_length)

        result = 0 if is_draw else 1
        feed.append(fbg.feature(), prob_leaf, result)

        for j in range(fbg.next_num):
            id_ = -(j + 1)
            feed._result[id_] = result
            result = -result

    print("")

    accuracy = float(correct_cnt) / play_cnt * 100  # percent
    ave_length = float(sum(lengths)) / len(lengths)

    print ("match: accuracy=%.1f[%%] average length=%.1f" % (
        accuracy, ave_length))
    log_file = open("match_log.txt", "a")
    log_file.write("%.2f\t%.2f\n" % (accuracy, ave_length))
    log_file.close()

    return accuracy


def test_match(use_gpu, search_limit, initial_life=1, reuse=False, show_info=False):
    # print test game

    print("\n<test game>")
    ckpt_path = "ckpt/model"
    tree = search.Tree(ckpt_path, use_gpu, 0, reuse=reuse)
    fbg = FizzBuzzGame(initial_life)

    continue_game = True

    while(continue_game):
        num = tree.search(fbg, search_limit,
                          use_dirichlet=False, show_info=show_info)
        continue_game = fbg.play(num)

    fbg.print_record()
