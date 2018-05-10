# -*- coding: utf-8 -*-

import numpy as np

PLAYER_NB = 2
DRAW_NB = 100
FEATURE_SIZE = 4   # numeric, Fizz, Buzz and FizzBuzz
HISTORY_SIZE = 10  # maximum size players can remember


def num2str(num):
    # convert number to string

    if DRAW_NB < num and num <= DRAW_NB + 3:
        return ("Fizz", "Buzz", "FizzBuzz")[num - 1 - DRAW_NB]

    return str(num)


class FizzBuzzGame(object):
    # class of FizzBuzz game
    #
    #  member variable
    #    record   : full game record from start to the present
    #    next_num : next number to be answered if not Fizz/Buzz/FizzBuzz.
    #    player_id: player index to answer next
    #    lives    : number of remaining life of each player

    def __init__(self, initial_life_=1):
        # initialize the game

        self.record = np.full((DRAW_NB), -1)
        self.next_num = 1  # start from 1
        self.player_id = 0  # start from 0
        self.lives = np.full((PLAYER_NB), initial_life_)
        self.initial_life = initial_life_

    def clear(self):
        # reset the game

        self.record.fill(-1)
        self.next_num = 1
        self.player_id = 0
        self.lives.fill(self.initial_life)

    def copy_to(self, target_):
        # copy to the target fbg

        target_.record = np.copy(self.record)
        target_.next_num = self.next_num
        target_.player_id = self.player_id
        target_.lives = np.copy(self.lives)
        target_.initial_life = self.initial_life

    def legal(self, num, next_num):
        # return whether input number is legal

        if next_num % 15 == 0:
            return num == DRAW_NB + 3  # FizzBuzz
        elif next_num % 3 == 0:
            return num == DRAW_NB + 1  # Fizz
        elif next_num % 5 == 0:
            return num == DRAW_NB + 2  # Buzz
        else:
            return num == next_num     # numeric

    def competitors_alive(self):
        # return whether at least 2 players survive

        return np.count_nonzero(self.lives) > 1

    def reach_limit(self):
        # return whether the game reaches the maximum numbers.
        # this is not corresponding to draw because the last
        # player might answer illegal number.

        return self.next_num > DRAW_NB

    def play(self, num):
        # add a number and return whether the game continues.
        # number must be within 1-100 for numeric or 101-103
        # for Fizz/Buzz/FizzBuzz.

        if self.reach_limit():
            return False

        if not self.legal(num, self.next_num):
            self.lives[self.player_id] -= 1

        self.record[self.next_num - 1] = num
        self.next_num += 1

        for _ in range(PLAYER_NB):
            inc_id = self.player_id + 1
            self.player_id = 0 if inc_id >= PLAYER_NB else inc_id

            if self.lives[self.player_id] > 0:
                break

        return (not self.reach_limit()) and self.competitors_alive()

    def feature(self):
        # return input features for the neural network

        f_ = np.zeros((HISTORY_SIZE, FEATURE_SIZE), dtype=np.float)

        for i in range(HISTORY_SIZE):

            idx_ = self.next_num - 1 - HISTORY_SIZE + i
            if idx_ < 0 or self.record[idx_] < 0 or self.record[idx_] > DRAW_NB + 3:
                continue  # all features are zero
            elif self.record[idx_] > DRAW_NB:
                # Fizz, Buzz or FizzBuzz
                f_[i][self.record[idx_] - DRAW_NB] = 1
            elif self.record[idx_] == idx_ + 1:
                f_[i][0] = 1  # numeric

        return f_

    def hash(self):
        # return hash of game record

        return (hash(self.record.tostring()) ^ self.player_id)

    def print_record(self):
        # print game record

        str_ = "player lives = ("
        for i in range(PLAYER_NB):
            str_ += str(self.lives[i])
            str_ += ", " if i < PLAYER_NB - 1 else ")\n"

        str_ += "game record ([]=wrong answer):\n\n"
        for i in range(self.next_num - 1):

            str_num = num2str(self.record[i])
            if not self.legal(self.record[i], i + 1):
                str_num = "[" + str_num + "]"
            str_ += str_num

            if i != self.next_num - 2:
                str_ += ", "
            if (i + 1) % 10 == 0:
                str_ += "\n"

        print(str_)
