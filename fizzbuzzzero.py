#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import learn
import match


if __name__ == "__main__":
    # suppress tensorflow warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # argument parser
    parser = argparse.ArgumentParser(
        description="Learning FizzBuzz without human knowledge")
    parser.add_argument("--learn", action="store_true",
                        help="start to learn")
    parser.add_argument("--game_cnt", type=int,
                        default=100, help="games to be played in an epoch")
    parser.add_argument("--search_limit", type=int,
                        default=100, help="limit of search count")
    parser.add_argument("--initial_life", type=int,
                        default=1, help="initial life of each player")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show thinking log of the first game")

    # not need to use GPU because there is no difference in speed
    parser.add_argument("--gpu", action="store_true",
                        help="enable to use GPU for learning")
    parser.add_argument("--gpu_cnt", type=int,
                        default=1, help="number of GPUs used for learning")

    args = parser.parse_args()

    if args.learn:
        # start learning

        ckpt_path = ""
        acc_list = []
        terminate_list = [100.0 for _ in range(3)]
        feed = match.Feed()

        for i in range(100):

            print("%d total games / next epoch: %d " %
                  (i * args.game_cnt, i + 1))

            acc = match.feed_match(feed, args.game_cnt, args.search_limit, ckpt_path,
                                   args.initial_life, use_gpu=args.gpu, gpu_idx=0,
                                   reuse=(i != 0), show_info=args.verbose)

            acc_list.append(acc)
            if len(acc_list) >= 3 and acc_list[-3:] == terminate_list:
                print("\naccuracy seems to be stable at 100%")
                break

            fp = match.FeedPicker(feed)
            learn.learn(fp, ckpt_path, 1e-4, use_gpu=args.gpu,
                        gpu_cnt=args.gpu_cnt)
            ckpt_path = "ckpt/model"

    # check if ckpt files exists
    if glob.glob("ckpt/*.data*") == []:
        print("ckpt files not found.")
        print("use \'--learn\' option to start learning or copy files from \'pre-train/ckpt\' to \'ckpt\' directory.")
        exit(0)

    # show game record
    match.test_match(args.gpu, args.search_limit, args.initial_life,
                     reuse=args.learn, show_info=args.verbose)
