
import glob
import os

import sys

import nltk

from nltk import *

import math


def ngram_prob(token_list, dist, dist_minus1):

    list_length = len(token_list)

    sliced_list = token_list[0:list_length-1]

    

    if(dist_minus1.freq(tuple(sliced_list)) == 0):

        return 0

    return math.log(dist.freq(tuple(token_list))/dist_minus1.freq(tuple(sliced_list)))

    



def docprob(token, n, dist, dist_minus1):

    logprob = 0

    for i in range(n-1, len(token)):

        token_list = []

        for a in range (i-n+1, i+1):

            token_list.append(token[a])


        print(token_list)

        print("\n")
        ngram_prob(token_list, dist, dist_minus1)




def main():

    train_path = sys.argv[1]

    dev_path = sys.argv[2]

    for train_filename in glob.glob(os.path.join(train_path, "*.tra")):
        f = open (train_filename, "r")


        contents = f.read()

        token = nltk.word_tokenize(contents)

        for n in range (2,4):
            
            ngram = ngrams(token,n)

            dist = nltk.FreqDist(ngram)

            dist_minus1 = None

            if (n>1):

                n_minus1_gram = ngrams(token,n-1)

                dist_minus1 = nltk.FreqDist(n_minus1_gram)

            

            for dev_filename in glob.glob(os.path.join(dev_path, "*.dev")):
                f = open (dev_filename, "r")
                contents = f.read()
                
                token = nltk.word_tokenize(contents)



                break

            docprob(token, n, dist, dist_minus1)


       



if __name__ == "__main__":
    main()





