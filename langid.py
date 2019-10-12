import argparse
import collections
import csv
import glob
import math

from nltk import *


class LanguageModel:

    def __init__(self, filename_train, n_value, ngram, n_minus1_gram, tokens_train, vocabulary):
        self.filename_train = filename_train

        self.n_value = n_value

        self.ngram = ngram
        self.n_minus1_gram = n_minus1_gram

        self.tokens_train = tokens_train

        self.vocabulary = vocabulary  # used for Laplace smoothing


class InterpolationLanguageModel:
    def __init__(self, filename_train, n_value, tokens_train, gram_list, lambda_value_list):
        self.filename_train = filename_train
        self.tokens_train = tokens_train
        self.n_value = n_value
        self.gram_list = gram_list
        self.lambda_value_list = lambda_value_list


def training_language_models(path_train):
    language_models = []

    for filename_train in glob.glob(os.path.join(path_train, "*")):

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

        for n in range(1, 7):

            ngram = ngrams(tokens_train, n)

            ngram = list(ngram)

            if (n == 1):
                n_minus1_gram = None

            else:
                n_minus1_gram = ngrams(tokens_train, n - 1)

                n_minus1_gram = list(n_minus1_gram)

            temp_lang_model = LanguageModel(filename_train, n, ngram, n_minus1_gram, tokens_train, vocabulary)
            language_models.append(temp_lang_model)

    return language_models


def training_interpolation_language_models(path_train, n):
    language_models = []

    for filename_train in glob.glob(os.path.join(path_train, "*")):

        f_train = open(filename_train, "r")
        contents_train = f_train.read()
        tokens_train = list(contents_train)

        gram_list = [[]]  # setting first element to empty list so that we can put ith gram in index i

        for i in range(1, n + 1):
            igram = ngrams(tokens_train, i)
            igram = list(igram)

            gram_list.append(igram)

        lambda_value_list = get_lambda_values(n, gram_list)  # ith lambda is in index i and so on, lambda 1 in index 1
        print(lambda_value_list)

        temp_lang_model = InterpolationLanguageModel(filename_train, n, tokens_train, gram_list,
                                                     lambda_value_list)
        language_models.append(temp_lang_model)

    return language_models


def write_to_file(output_filename, output_list):
    with open(output_filename, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')

        for output in output_list:
            tsv_writer.writerow([output[0], output[1], output[2], output[3]])


def ngram_prob_laplace(token_list, ngram, n_minus1_gram, vocabulary):
    list_length = len(token_list)

    sliced_list = token_list[0:list_length - 1]

    numerator = ngram.count(tuple(token_list)) + 1

    denominator = n_minus1_gram.count(tuple(sliced_list)) + vocabulary

    return math.log2(numerator / denominator)


def docprob_laplace(token, n, ngram, n_minus1_gram, vocabulary):
    logprob = 0
    for i in range(n - 1, len(token)):

        token_list = []

        for a in range(i - n + 1, i + 1):
            token_list.append(token[a])

        logprob += ngram_prob_laplace(token_list, ngram, n_minus1_gram, vocabulary)
    return logprob


def docprob_unigram(tokens_test, tokens_train):
    logprob = 0

    for i in range(0, len(tokens_test)):

        if tokens_train.count(tokens_test[i]) == 0:
            tokens_test[i] = "#"

        logprob += math.log2(tokens_train.count(tokens_test[i]) / len(tokens_train))
    return logprob


def calculate_interpolation_probability(gram_list, tokens_test, lambda_value_list, n):
    ngram_test = list(ngrams(tokens_test, n))

    total_log_prob = 0
    for each_tuple in ngram_test:

        prob = 0
        for i in range(1, n + 1):

            if i == n:
                numerator = gram_list[1].count(tuple(each_tuple[n - 1]))
                denominator = len(gram_list[1])

            else:
                n_minus_i_plus_1_gram = gram_list[n - i + 1]
                n_minus_i_gram = gram_list[n - i]

                numerator = n_minus_i_plus_1_gram.count(each_tuple[i - 1:n])
                denominator = n_minus_i_gram.count(each_tuple[0:n - i])

            if denominator == 0:
                prob += 0
            else:
                prob += lambda_value_list[i] * numerator / denominator

        if prob == 0:
            total_log_prob += 0
        else:
            total_log_prob += math.log2(prob)

    return total_log_prob


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


def get_lambda_values(n, gram_list):
    lambda_value_list = [0]  # setting first element 0 so that we can put value of ith lambda in index i

    for i in range(1, n + 1):
        lambda_value_list.append(0)

    if len(gram_list) == n + 1:
        ngram = gram_list[n]

        for each_tuple in ngram:

            max_count = - sys.maxsize - 1
            max_count_index = 0

            ngram_each_tuple_count = ngram.count(each_tuple)

            if ngram_each_tuple_count > 0:

                for i in range(1, n + 1):

                    if i == n:
                        numerator = gram_list[1].count(tuple(each_tuple[n - 1])) - 1
                        denominator = len(gram_list[1]) - 1

                    else:
                        n_minus_i_plus_1_gram = gram_list[n - i + 1]
                        n_minus_i_gram = gram_list[n - i]

                        numerator = n_minus_i_plus_1_gram.count(each_tuple[i - 1:n]) - 1
                        denominator = n_minus_i_gram.count(each_tuple[0:n - i]) - 1

                    if denominator == 0:
                        count = 0
                    else:
                        count = numerator / denominator

                    if count > max_count:
                        max_count = count
                        max_count_index = n - i + 1

                lambda_value_list[max_count_index] += ngram_each_tuple_count

        normalized_lambda_value_list = normalize_lambda_values(lambda_value_list)

        return normalized_lambda_value_list

    else:
        return lambda_value_list


def unsmoothed_model(n, language_models, tokens_test, filename_test):
    min_perplexity = sys.maxsize

    best_guess_train_file = None

    for language_model in language_models:

        if language_model.n_value == 1:

            logprob = docprob_unigram(tokens_test, language_model.tokens_train)
            perplexity = 2 ** -(logprob / len(tokens_test))

            if perplexity < min_perplexity:
                min_perplexity = perplexity

                best_guess_train_file = language_model.filename_train

            # print(filename_test, best_guess_train_file)

    output_line = []

    output_line.append(filename_test)
    output_line.append(best_guess_train_file)
    output_line.append(min_perplexity)
    output_line.append(n)

    return output_line
    # print(compare_file_names_ignoring_extension(filename_test, best_guess_train_file))


def laplace_model(n, language_models, tokens_test, filename_test):
    min_perplexity = sys.maxsize

    best_guess_train_file = None

    for language_model in language_models:

        if language_model.n_value == n:

            logprob = docprob_laplace(tokens_test, n, language_model.ngram, language_model.n_minus1_gram,
                                      language_model.vocabulary)

            perplexity = 2 ** -(logprob / len(tokens_test))

            if perplexity < min_perplexity:
                min_perplexity = perplexity
                best_guess_train_file = language_model.filename_train

    output_line = []

    output_line.append(filename_test)
    output_line.append(best_guess_train_file)
    output_line.append(min_perplexity)
    output_line.append(n)

    return output_line


def interpolation_model(language_models, tokens_test, filename_test, n):
    min_perplexity = sys.maxsize

    best_guess_train_file = None

    for language_model in language_models:

        if language_model.n_value == n:

            log_docprob = calculate_interpolation_probability(language_model.gram_list, tokens_test,
                                                              language_model.lambda_value_list, n)

            perplexity = 2 ** -(log_docprob / len(tokens_test))

            if perplexity < min_perplexity:
                min_perplexity = perplexity
                best_guess_train_file = language_model.filename_train

    output_line = []

    output_line.append(filename_test)
    output_line.append(best_guess_train_file)
    output_line.append(min_perplexity)
    output_line.append(n)

    return output_line


# usage: langid.py [-h] --train TRAIN_PATH --test TEST_PATH [--unsmoothed] [--laplace] [--interpolation]
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest="train_path", action="store", default="811_a1_train", required=True)
    parser.add_argument('--test', dest="test_path", action="store", default="811_a1_test_final", required=True)
    parser.add_argument('--unsmoothed', action="store_true", default=False)
    parser.add_argument('--laplace', action="store_true", default=False)
    parser.add_argument('--interpolation', action="store_true", default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    path_train = args.train_path

    path_test = args.test_path

    value_of_n = 8

    if args.laplace:
        language_models = training_language_models(path_train)
        output_filename = "results_test_laplace.txt"

    elif args.interpolation:
        language_models = training_interpolation_language_models(path_train, value_of_n)
        output_filename = "results_test_interpolation.txt"

    else:
        language_models = training_language_models(path_train)
        output_filename = "results_test_unsmoothed.txt"

    output_list = []

    perp = 0
    count = 0

    for filename_test in sorted(glob.glob(os.path.join(path_test, "*"))):

        f_test = open(filename_test, "r")
        contents_test = f_test.read()
        tokens_test = list(contents_test)

        if args.laplace:
            output_line = laplace_model(value_of_n, language_models, tokens_test, filename_test)

            """if(compare_file_names_ignoring_extension(output_line[0], output_line[1])):

            	count += 1 ;

            	perp += output_line[2]"""

        elif args.interpolation:
            output_line = interpolation_model(language_models, tokens_test, filename_test, value_of_n)
            print(output_line[0], output_line[1])
            if compare_file_names_ignoring_extension(output_line[0], output_line[1]):
                count += 1

                perp += output_line[2]

        else:
            output_line = unsmoothed_model(1, language_models, tokens_test, filename_test)

        output_list.append(output_line)

    write_to_file(output_filename, output_list)

    print(count)

    print(perp / count)


if __name__ == "__main__":
    main()
