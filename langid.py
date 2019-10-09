import glob
import os
import argparse
import sys

import nltk

from nltk import *

import math


class Lang_Model:

    def __init__(self, filename_train, n_value, dist, dist_minus1, token_count):
        self.filename_train = filename_train
        self.n_value = n_value
        self.dist = dist
        self.dist_minus1 = dist_minus1

        self.token_count = token_count  # need this for unigram only


def training_language_models(path_train):
    language_models = []

    for filename_train in glob.glob(os.path.join(path_train, "*.tra")):

        f_train = open(filename_train, "r")
        contents_train = f_train.read()
        tokens_train = list(contents_train)

        for n in range(1, 10):

            ngram = ngrams(tokens_train, n)
            dist = nltk.FreqDist(ngram)

            if (n == 1):
                dist_minus1 = None

            else:
                n_minus1_gram = ngrams(tokens_train, n - 1)
                dist_minus1 = nltk.FreqDist(n_minus1_gram)

            temp_lang_model = Lang_Model(filename_train, n, dist, dist_minus1, len(tokens_train))
            language_models.append(temp_lang_model)

    return language_models


def ngram_prob(token_list, dist, dist_minus1):
    list_length = len(token_list)

    sliced_list = token_list[0:list_length - 1]

    if dist.freq(tuple(token_list)) == 0 or dist_minus1.freq(tuple(sliced_list)) == 0:
        return -7

    return math.log2(dist.freq(tuple(token_list)) / dist_minus1.freq(tuple(sliced_list)))


def docprob_unigram(token_dev, dist, length_token_train):
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

        logprob += ngram_prob(token_list, dist, dist_minus1)
    return logprob


# returns True if two filenames excluding extension are same
# returns False if two filenames excluding extension are different
def compare_file_names_ignoring_extension(filename1, filename2):
    # gets the file name as "udhr-yao" from "811_a1_dev\udhr-yao.txt.dev" and "811_a1_train\udhr-yao.txt.tra"
    filename_prefix_pattern = r"\\(.*?)\..*"

    file1_prefix = re.findall(filename_prefix_pattern, filename1)
    file2_prefix = re.findall(filename_prefix_pattern, filename2)

    return len(file1_prefix) != 0 and len(file2_prefix) != 0 and file1_prefix[0] == file2_prefix[0]


# usage: langid.py [-h] --train TRAIN_PATH --dev DEV_PATH [--unsmoothed] [--laplace] [--interpolation]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest="train_path", action="store", default="811_a1_train", required=True)
    parser.add_argument('--dev', dest="dev_path", action="store", default="811_a1_dev", required=True)
    parser.add_argument('--unsmoothed', action="store_true", default=False)
    parser.add_argument('--laplace', action="store_true", default=False)
    parser.add_argument('--interpolation', action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_arguments()

    path_train = args.train_path

    path_dev = args.dev_path

    # if args.unsmoothed:
    #     # apply unsmoothing
    #
    # if args.laplace:
    #     # apply laplace smoothing
    #
    # if args.interpolation:
    #     # apply interpolation smoothing

    language_models = training_language_models(path_train)

    for n in range(1, 10):

        for filename_dev in glob.glob(os.path.join(path_dev, "*.dev")):

            min_perplexity = sys.maxsize

            best_guess_train_file = None

            f_dev = open(filename_dev, "r")
            contents_dev = f_dev.read()
            token_dev = list(contents_dev)

            for language_model in language_models:

                if language_model.n_value == n:

                    if n == 1:

                        logprob = docprob_unigram(token_dev, language_model.dist, language_model.token_count)

                    else:

                        logprob = docprob(token_dev, n, language_model.dist, language_model.dist_minus1)

                    perplexity = 2 ** -(logprob / len(token_dev))

                    if perplexity < min_perplexity:
                        min_perplexity = perplexity

                        best_guess_train_file = language_model.filename_train

            print(filename_dev, best_guess_train_file)
            print(compare_file_names_ignoring_extension(filename_dev, best_guess_train_file))


if __name__ == "__main__":
    main()
