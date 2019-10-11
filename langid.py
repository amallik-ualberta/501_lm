import glob
import os
import argparse
import sys

import nltk

from nltk import *

import math

import collections

import csv

class Lang_Model:

    def __init__(self, filename_train, n_value, ngram, n_minus1_gram, tokens_train, vocabulary):
        self.filename_train = filename_train

        self.n_value = n_value

        self.ngram = ngram
        self.n_minus1_gram = n_minus1_gram

        self.tokens_train = tokens_train  

        self.vocabulary = vocabulary  # used for Laplace smoothing


def training_language_models(path_train):
    language_models = []

    for filename_train in glob.glob(os.path.join(path_train, "*.tra")):

        f_train = open(filename_train, "r")
        contents_train = f_train.read()
        tokens_train = list(contents_train)

        groups_to_replace = 2  # how many group of characters you want to replace with 'UNK'?

        counter = collections.Counter(tokens_train)
        least_commons = counter.most_common()[:-groups_to_replace - 1:-1]

        for least_common in least_commons:

            character = least_common[0]

            for n, i in enumerate(tokens_train):

                if (i == character):
                    tokens_train[n] = "#"

        unique_tokens = set(tokens_train)

        vocabulary = len(unique_tokens)

        for n in range(1, 10):

            ngram = ngrams(tokens_train, n)

            ngram = list(ngram)

            if (n == 1):
                n_minus1_gram = None

            else:
                n_minus1_gram = ngrams(tokens_train, n - 1)

                n_minus1_gram = list(n_minus1_gram)

            temp_lang_model = Lang_Model(filename_train, n, ngram, n_minus1_gram, tokens_train, vocabulary)
            language_models.append(temp_lang_model)

    return language_models



def write_to_file (output_filename, output_list):

	with open (output_filename, 'wt') as out_file:

		tsv_writer = csv.writer(out_file, delimiter = '\t')

		for output in output_list:

			tsv_writer.writerow([output[0],output[1],output[2],output[3]])



def ngram_prob(token_list, ngram, n_minus1_gram, vocabulary):
    list_length = len(token_list)

    sliced_list = token_list[0:list_length - 1]

    numerator = ngram.count(tuple(token_list)) + 1

    denominator = n_minus1_gram.count(tuple(sliced_list)) + vocabulary

    return math.log2(numerator / denominator)





def docprob(token, n, ngram, n_minus1_gram, vocabulary):
    logprob = 0
    for i in range(n - 1, len(token)):

        token_list = []

        for a in range(i - n + 1, i + 1):
            token_list.append(token[a])

        logprob += ngram_prob(token_list, ngram, n_minus1_gram, vocabulary)
    return logprob


def docprob_unigram(tokens_dev, tokens_train):
    logprob = 0

    for i in range(0, len(tokens_dev)):

        if tokens_train.count(tokens_dev[i]) == 0:
            tokens_dev[i] = "#"

        logprob += math.log2(tokens_train.count(tokens_dev[i]) / len(tokens_train))
    return logprob

# returns True if two filenames excluding extension are same
# returns False if two filenames excluding extension are different
def compare_file_names_ignoring_extension(filename1, filename2):
    # gets the file name as "udhr-yao" from "811_a1_dev\udhr-yao.txt.dev" and "811_a1_train\udhr-yao.txt.tra" for windows
    # gets the file name as "udhr-yao" from "811_a1_dev/udhr-yao.txt.dev" and "811_a1_train/udhr-yao.txt.tra" for unix
    filename_prefix_pattern = r"[/\\](.*?)\..*"

    file1_prefix = re.findall(filename_prefix_pattern, filename1)
    file2_prefix = re.findall(filename_prefix_pattern, filename2)

    return len(file1_prefix) != 0 and len(file2_prefix) != 0 and file1_prefix[0] == file2_prefix[0]


def normalize_lambda_values(lambda_value_list):
    sum_val = 0
    for value in lambda_value_list:
        sum_val += value

    for i in range(1, len(lambda_value_list)):
        lambda_value_list[i] = lambda_value_list[i] / sum_val

    return lambda_value_list


def get_lambda_values(n, tokens_train):
    gram_list = [[]]  # setting first element to empty list so that we can put ith gram in index i

    lambda_value_list = [0]  # setting first element 0 so that we can put value of ith lambda in index i

    for i in range(1, n + 1):
        igram = ngrams(tokens_train, i)
        igram = list(igram)

        gram_list.append(igram)
        lambda_value_list.append(0)

    ngram = gram_list[n]

    for each_tuple in ngram:

        max_count = - sys.maxsize - 1
        max_count_index = 0

        ngram_each_tuple_count = ngram.count(each_tuple)

        if ngram_each_tuple_count > 0:

            for i in range(1, n + 1):

                if i == n:

                    if len(gram_list[1]) - 1 == 0:
                        count = 0
                    else:
                        count = (gram_list[1].count(tuple(each_tuple[n - 1])) - 1) / (len(gram_list[1]) - 1)

                else:
                    n_minus_i_plus_1_gram = gram_list[n - i + 1]
                    n_minus_i_gram = gram_list[n - i]

                    if n_minus_i_gram.count(each_tuple[0:n - i]) - 1 == 0:
                        count = 0
                    else:
                        count = (n_minus_i_plus_1_gram.count(each_tuple[i - 1:n]) - 1) / (
                                n_minus_i_gram.count(each_tuple[0:n - i]) - 1)

                if count > max_count:
                    max_count = count
                    max_count_index = n - i + 1

            lambda_value_list[max_count_index] += ngram_each_tuple_count

    normalized_lambda_value_list = normalize_lambda_values(lambda_value_list)

    return normalized_lambda_value_list



def unsmoothed_model(n, language_models, tokens_dev, filename_dev):

    


    min_perplexity = sys.maxsize

    best_guess_train_file = None

    for language_model in language_models:

        if language_model.n_value == 1:

            logprob = docprob_unigram(tokens_dev, language_model.tokens_train)
            perplexity = 2 ** -(logprob / len(tokens_dev))

            if perplexity < min_perplexity:
                min_perplexity = perplexity

                best_guess_train_file = language_model.filename_train

            # print(filename_dev, best_guess_train_file)

    output_line = []

    output_line.append(filename_dev)
    output_line.append(best_guess_train_file)
    output_line.append(min_perplexity)
    output_line.append(n)

    return output_line
    #print(compare_file_names_ignoring_extension(filename_dev, best_guess_train_file))


def laplace_model (n, language_models, tokens_dev, filename_dev):

	min_perplexity = sys.maxsize

	best_guess_train_file = None

	for language_model in language_models:

	    if language_model.n_value == n:

	        logprob = docprob(tokens_dev, n, language_model.ngram, language_model.n_minus1_gram,
	                                          language_model.vocabulary)

	        perplexity = 2 ** -(logprob / len(tokens_dev))

	        if perplexity < min_perplexity:
	            min_perplexity = perplexity
	            best_guess_train_file = language_model.filename_train

	output_line = []

	output_line.append(filename_dev)
	output_line.append(best_guess_train_file)
	output_line.append(min_perplexity)
	output_line.append(n)

	return output_line

	#print(filename_dev, best_guess_train_file)


def interpolation_model(n, language_models, tokens_dev, filename_dev):

    for language_model in language_models:

        if language_model.n_value == n:

            lambda_list = get_lambda_values(n, language_model.tokens_train)
            print(lambda_list)








# usage: langid.py [-h] --train TRAIN_PATH --dev DEV_PATH [--unsmoothed] [--laplace] [--interpolation]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest="train_path", action="store", default="811_a1_train", required=True)
    parser.add_argument('--dev', dest="dev_path", action="store", default="811_a1_dev", required=True)
    parser.add_argument('--unsmoothed', action="store_true", default=False)
    parser.add_argument('--laplace', action="store_true", default=False)
    parser.add_argument('--interpolation', action="store_true", default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    path_train = args.train_path

    path_dev = args.dev_path

    language_models = training_language_models(path_train)

    output_list = []
    

        
    perp = 0;
    count = 0;

    for filename_dev in glob.glob(os.path.join(path_dev, "*.dev")):


        f_dev = open(filename_dev, "r")
        contents_dev = f_dev.read()
        tokens_dev = list(contents_dev)

        if args.unsmoothed:

            output_line = unsmoothed_model(1, language_models, tokens_dev, filename_dev)


        if args.laplace:

            output_line = laplace_model(7, language_models, tokens_dev, filename_dev)

            """if(compare_file_names_ignoring_extension(output_line[0], output_line[1])):

            	count += 1 ;

            	perp += output_line[2]"""



            
        if args.interpolation:

            interpolation_model(3, language_models, tokens_dev, filename_dev)

            output_line = [] #Write code to get ouput line. output line contains 4 things. see laplace method above.


        output_list.append(output_line)

    if args.unsmoothed:

    	output_filename = "results_dev_unsmoothed.txt"

    if args.laplace:
        output_filename = "results_dev_laplace.txt"

    if args.interpolation:

    	output_filename = "results_dev_interpolation.txt"

    write_to_file(output_filename, output_list)

    print(count)

    print(perp/count)




if __name__ == "__main__":
    main()
