Team Members:
    Arnob Mallik (CCID: amallik)
    Arif Hasnat (CCID: hasnat)

Execution Instruction:
    run the following command in command line:
        python langid.py --train TRAIN_FOLDER_PATH --test TEST_FOLDER_PATH [OPTIONS}

        OPTIONS: --unsmoothed
                 --laplace
                 --interpolation
    example:
        python langid.py --train 811_a1_train --test 811_a1_test_final --unsmoothed