Team Members:
    Arnob Mallik (CCID: amallik)
    Arif Hasnat (CCID: hasnat)

Execution Instruction:
    run the following command in command line:
        python langid.py --train TRAIN_FOLDER_PATH --dev DEV_FOLDER_PATH [OPTIONS}

        OPTIONS: --unsmoothed
                 --laplace
                 --interpolation
    example:
        python langid.py --train 811_a1_train --dev 811_a1_dev --unsmoothed