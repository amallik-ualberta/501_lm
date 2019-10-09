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
        return -7

    return math.log2(dist.freq(tuple(token_list)) / dist_minus1.freq(tuple(sliced_list)))


def docprob_unigram(token_dev, dist, length_token_train):

    print (len(dist), length_token_train)
    logprob = 0

    for i in range(0, len(token_dev)):
        if dist.freq(tuple(token_dev[i])) == 0:
            logprob += -7
        else:
            logprob += math.log2(dist.freq(tuple(token_dev[i])) / length_token_train)
    return logprob


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


class Output:
    dev_filename = None
    train_filename = None
    n = 0
    perplexity = sys.maxsize


def main():
    path_train = sys.argv[1]

    path_dev = sys.argv[2]

    for filename_dev in glob.glob(os.path.join(path_dev, "*.dev")):
        f_dev = open(filename_dev, "r")
        contents_dev = f_dev.read()

        # token_dev = nltk.word_tokenize(contents_dev)
        token_dev = list(contents_dev)
        output = Output()
        for n in range(1, 9):
            for filename_train in glob.glob(os.path.join(path_train, "*.tra")):
                f_train = open(filename_train, "r")

                contents_train = f_train.read()

                # token_train = nltk.word_tokenize(contents_train)
                token_train = list(contents_train)

                ngram = ngrams(token_train, n)
                dist = nltk.FreqDist(ngram)

                if n == 1:
                    logprob = docprob_unigram(token_dev, dist, len(token_train))
                else:
                    n_minus1_gram = ngrams(token_train, n - 1)

                    dist_minus1 = nltk.FreqDist(n_minus1_gram)
                    logprob = docprob(token_dev, n, dist, dist_minus1)

                perplexity = 2 ** -(logprob / len(token_dev))
                # print("dev file: %s, train file: %s, value of n: %d, perplexity: %s" % (
                #     filename_dev, filename_train, n, perplexity))
                if perplexity < output.perplexity:
                    output.dev_filename = filename_dev
                    output.train_filename = filename_train
                    output.n = n
                    output.perplexity = perplexity
        print("best result -> dev file: %s, train file: %s, value of n: %d, perplexity: %s" % (
            output.dev_filename, output.train_filename, output.n, output.perplexity))
        break


if __name__ == "__main__":
    main()
