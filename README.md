# Cross-Domain_Gender_Identification_ML_Codebase

When using code from this repository, please credit Qidi J. Wang and Alec G. Moore.

This repository contains the codebase for replicating the machine learning experiments in the paper ["Cross-Domain Gender Identification Using VR Tracking Data"](https://doi.org/10.1109/ISMAR62088.2024.00032) by Wang et al.

We recommend using a Jupyter Notebook to import the codebase and call the functions used to run experiments.

Instructions:

1. Import ml_codebase

2. Call the functions getRawFAST and getRawAlyx to read from the csv data files

    ex. fast_a_raw, fast_a_path = ml_codebase.getRawFAST(fast_a_folder_filepath)

3. You may want to pickle fast_a_raw and other variables by using the function 'cucumberIntoJar' to save time for future experiments. This way, you can simply unpickle the data instead of having to read the files in again.

    ex. ml_codebase.cucumberIntoJar(fast_a_raw,pickle_filename)

4. We defined our runExperiment functions to work so that the machine learning models would train once on the data specified in train_info, and then test on the data specified in test1_info, test2_info, and test3_info. This saves time by avoiding training on the same data 3 times. For example, calling runExperimentsScene one time can produce the experiment results for train/test pairs: FAST A -> FAST B, FAST A -> Alyx A, and FAST A -> Alyx B.

   The experiments require certain parameters. To save space, specify them in dictionaries of the below specified formats:

    - train_info = {'raw': var, 'path': var, 'fora': '', 'set': ''}
    - test1_info = {'raw': var, 'path': var, 'fora': '', 'set': '', 'filename': ''}
    - test2_info = {'raw': var, 'path': var, 'fora': '', 'set': '', 'filename': ''}
    - test3_info = {'raw': var, 'path': var, 'fora': '', 'set': '', 'filename': ''}

    Here is an explanation of each key of the dictionary:


    - 'raw' is the variable containing the raw data, like fast_a_raw
    - 'path' is the variable containing the path to the folder containing the data
    - 'fora' is short for 'fast or alyx'. This should be a string: either 'FAST' or 'Alyx' to ensure the corresponding functions are called correctly
    - 'set' is a string to describe the name of the set you are using. This is used for consistent naming conventions. The options are typically 'FASTA', 'FASTB', 'AlyxA', and 'AlyxB'
    - 'filename' is also a string for the purpose of naming conventions. This should describe the experiment you are conducting. The format is typically 'trainingset-testingset'. See example below.

    ex. train_info = {'raw': fast_a_raw, 'path': fast_a_path, 'fora': 'FAST', 'set': 'FASTA'}
  
    ex. test1_info = {'raw': fast_b_raw, 'path': fast_b_path, 'fora': 'FAST', 'set': 'FASTB', 'filename': 'FASTA-FASTB'}
  


5. After creating the dictionaries, call runExperiments to execute the experiments. 
    ex. ml_codebase.runExperiments(train_info, test1_info, test2_info, test3_info)

