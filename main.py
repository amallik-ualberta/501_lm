import glob
import os

import sys

import nltk

from nltk import *

import math


def ngram_prob(token_list, dist, dist_minus1):
    list_length = len(token_list)

    sliced_list = token_list[0:list_length - 1]

    if dist.freq(tuple(token_list)) == 0 or dist_minus1.freq(tuple(sliced_list)) == 0:
        return 0

    return math.log2(dist.freq(tuple(token_list)) / dist_minus1.freq(tuple(sliced_list)))


def docprob(token, n, dist, dist_minus1):
    logprob = 0
    for i in range(n - 1, len(token)):

        token_list = []

        for a in range(i - n + 1, i + 1):
            token_list.append(token[a])

        # print(token_list)

        # print("\n")
        logprob += ngram_prob(token_list, dist, dist_minus1)
    return logprob


def main():
    path_train = sys.argv[1]

    path_dev = sys.argv[2]

    for filename_dev in glob.glob(os.path.join(path_dev, "*.dev")):
        f_dev = open(filename_dev, "r")
        contents_dev = f_dev.read()

        # token_dev = nltk.word_tokenize(contents_dev)
        token_dev = list(contents_dev)

        for n in range(2, 9):
            for filename_train in glob.glob(os.path.join(path_train, "*.tra")):
                f_train = open(filename_train, "r")

                contents_train = f_train.read()

                # token_train = nltk.word_tokenize(contents_train)
                token_train = list(contents_train)

                ngram = ngrams(token_train, n)

                dist = nltk.FreqDist(ngram)

                dist_minus1 = None

                if n > 1:
                    n_minus1_gram = ngrams(token_train, n - 1)

                    dist_minus1 = nltk.FreqDist(n_minus1_gram)

                logprob = docprob(token_dev, n, dist, dist_minus1)
                perplexity = 2 ** -(logprob / len(token_dev))
                print("dev file: %s, train file: %s, value of n: %d, probability: %s, perplexity: %s" % (
                    filename_dev, filename_train, n, logprob, perplexity))
        break


if __name__ == "__main__":
    main()
