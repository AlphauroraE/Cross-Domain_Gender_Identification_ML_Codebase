# Import the necessary libraries and functions:
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from os import path
import pickle
from glob import glob
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# These are our ml models, with random_state = 0 for replicability (seed)
classifier = {
        'kNN': KNeighborsClassifier(), 
        'RF': RandomForestClassifier(random_state=0),
        'GBM': GradientBoostingClassifier(random_state=0)
        }


# Global variables:
features = []
# Set the path to the documents folder, or whatever folder you want the results to end up in
documents_filepath = 'C:\\Users\\username\\Documents'
response = ['Gender']
response_cutoff = [0.5]

"""Functions for Reading from Raw Data Files and Auxiliary Functions"""

def getRawFAST(path):
    PIDs = getPIDsFAST(path)
    raw_file = getFASTData(path,PIDs)
    return raw_file, path

def getRawAlyx(path):
    PIDs = getPIDsAlyx(path)
    raw_file = getAlyxData(path,PIDs)
    return raw_file, path


def getPIDsFAST(filepath):
    PIDretlist = []
    for file in os.listdir(filepath):
        if (file != ".DS_Store"):
            pid, x = userIDConventionFAST(filename=file)
            PIDretlist.append(pid)
    return PIDretlist

def getPIDsAlyx(filepath):
    PIDretlist = []
    for file in os.listdir(filepath):
        if (file != ".DS_Store"):
            pid, x = userIDConventionAlyx(filename=file)
            PIDretlist.append(pid)
    return PIDretlist


def convert_timestamp(value):
    value = float(value)
    datetime_value = pd.to_datetime(value, unit='s') 
    return datetime_value


def getFASTData(directory,PIDs=None):
    '''I originally thought this differs from getData in that no time_converter is needed bc the original full dataset has it 
    in proper format already, but since I found out we had to use the elapsed time instead for aggregation purposes, we still
    need it'''
    ret = dict()
    time_converter = {'Timestamp' : convert_timestamp}
    if (PIDs == None):
        print("PIDS can't be none!")
    else:
        for i in PIDs:
            ret[i] = pd.read_csv(glob(path.join(directory,f'FAST{i}*.csv'))[0],
                delimiter=',',parse_dates=['Timestamp'],converters=time_converter, index_col=['Timestamp'])
    return(ret)

def getAlyxData(directory,PIDs=None):
    '''getFASTData adapted for Alyx'''
    ret = dict()
    time_converter = {'Timestamp' : convert_timestamp}
    if (PIDs == None):
        print("PIDS can't be none!")
    else:
        for i in PIDs:
            ret[i] = pd.read_csv(glob(path.join(directory,f'Alyx{i}*.csv'))[0],
                delimiter=',',parse_dates=['Timestamp'],converters=time_converter, index_col=['Timestamp'])
    return(ret)


def userIDConventionFAST(filename = None,filepath = None):
    start_ind = 0
    end_ind = 0

    if (filename == None and filepath == None):
        print("Must provide filepath or filename")
    elif (filename == None):
        filename = Path(filepath).name

    for i in range(len(filename)):
        if (filename[i] == "F" and filename[i+1] == "A" and filename[i+2] == "B"):
            start_ind = i+3
        
    end_ind = start_ind + 5

    return filename[start_ind:end_ind],filename[end_ind] #2nd retval should be gender now, not number

def userIDConventionAlyx(filename = None,filepath = None):
    start_ind = 0
    end_ind = 0

    if (filename == None and filepath == None):
        print("Must provide filepath or filename")
    elif (filename == None):
        filename = Path(filepath).name


    for i in range(len(filename)):
        if (filename[i] == "A" and filename[i+1] == "l" and filename[i+2] == "y" and filename[i+3] == "x"):
            start_ind = i+4
        
    end_ind = start_ind + 2

    return filename[start_ind:end_ind],filename[end_ind] #2nd retval should be gender now, not number


"""Functions for Pickling/Unpickling"""

def cucumberIntoJar(cucumber,destinationName,path=None):
    global documents_filepath
    # Pickles the file
    if (path != None):
        file_path = path
    else:
        file_path = f'{documents_filepath}\\{destinationName}.pkl'

    with open(file_path, 'wb') as file:
        pickle.dump(cucumber,file)

    print("Pickled!")

def pickOutPickle(extractionpath):
    # Extracts the pickled file
    with open(extractionpath,'rb') as file:
        return(pickle.load(file))


"""Master Functions to Run Experiments"""

def runExperiments(train_info, test1_info, test2_info, test3_info):
    # train_info = {'raw': , 'path': , 'fora': , 'set': }
    # test1_info = {'raw': , 'path': , 'fora': , 'set': , 'filename': }
    import datetime
    global features
    global documents_filepath

    featsList = getFeatList()
    
    # Unpack train and test dictionaries
    df_train_raw = train_info['raw']
    TRAINpath = train_info['path']
    train_ForA = train_info['fora']
    train_set = train_info['set']

    df_test1_raw = test1_info['raw']
    TEST1path = test1_info['path']
    test1_ForA = test1_info['fora']
    test1_set = test1_info['set']
    outputfile1 = test1_info['filename']

    df_test2_raw = test2_info['raw']
    TEST2path = test2_info['path']
    test2_ForA = test2_info['fora']
    test2_set = test2_info['set']
    outputfile2 = test2_info['filename']

    df_test3_raw = test3_info['raw']
    TEST3path = test3_info['path']
    test3_ForA = test3_info['fora']
    test3_set = test3_info['set']
    outputfile3 = test3_info['filename']


    experiment_info1 = {'train_info': train_set, 'test_info': test1_set}
    experiment_info2 = {'train_info': train_set, 'test_info': test2_set}
    experiment_info3 = {'train_info': train_set, 'test_info': test3_set}



    # Folder names
    folder1_name = f"{train_set}-{test1_set}"
    folder2_name = f"{train_set}-{test2_set}"
    folder3_name = f"{train_set}-{test3_set}"
    
    # Make folder
    if (os.path.isdir(f"{documents_filepath}\\{folder1_name}") == False):
        os.mkdir(f"{documents_filepath}\\{folder1_name}")

    if (os.path.isdir(f"{documents_filepath}\\{folder2_name}") == False):
        os.mkdir(f"{documents_filepath}\\{folder2_name}")

    if (os.path.isdir(f"{documents_filepath}\\{folder3_name}") == False):
        os.mkdir(f"{documents_filepath}\\{folder3_name}")

    # Set up files and filewriters
    writer1 = open(f"{documents_filepath}\\{folder1_name}\\{outputfile1}.txt", "a")
    writer2 = open(f"{documents_filepath}\\{folder2_name}\\{outputfile2}.txt", "a")
    writer3 = open(f"{documents_filepath}\\{folder3_name}\\{outputfile3}.txt", "a")

    writer1.write(f"Train Path: {TRAINpath}\n")
    writer1.write(f"Test Path: {TEST1path}\n\n")

    writer2.write(f"Train Path: {TRAINpath}\n")
    writer2.write(f"Test Path: {TEST2path}\n\n")

    writer3.write(f"Train Path: {TRAINpath}\n")
    writer3.write(f"Test Path: {TEST3path}\n\n")


    # Establish score names
    score_name_knn_1 = f"{train_set}-{test1_set}_kNN"
    score_name_knn_2 = f"{train_set}-{test2_set}_kNN"
    score_name_knn_3 = f"{train_set}-{test3_set}_kNN"

    score_name_rf_1 = f"{train_set}-{test1_set}_RF"
    score_name_rf_2 = f"{train_set}-{test2_set}_RF"
    score_name_rf_3 = f"{train_set}-{test3_set}_RF"

    score_name_gbm_1 = f"{train_set}-{test1_set}_GBM"
    score_name_gbm_2 = f"{train_set}-{test2_set}_GBM"
    score_name_gbm_3 = f"{train_set}-{test3_set}_GBM"
    


    # Make score name folders
    # kNN
    if (os.path.isdir(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}") == False):
        os.mkdir(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}")

    if (os.path.isdir(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}") == False):
        os.mkdir(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}")

    if (os.path.isdir(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}") == False):
        os.mkdir(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}")

    # RF
    if (os.path.isdir(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}") == False):
        os.mkdir(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}")

    if (os.path.isdir(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}") == False):
        os.mkdir(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}")

    if (os.path.isdir(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}") == False):
        os.mkdir(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}")

    # GBM
    if (os.path.isdir(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}") == False):
        os.mkdir(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}")

    if (os.path.isdir(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}") == False):
        os.mkdir(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}")

    if (os.path.isdir(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}") == False):
        os.mkdir(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}")



    # Create all the score tables
    score_df_knn_1 = createScoreTable('kNN',experiment_info1,'A')
    f1_df_knn_1 = createScoreTable('kNN',experiment_info1,'F')
    prec_df_knn_1 = createScoreTable('kNN',experiment_info1,'P')
    recall_df_knn_1 = createScoreTable('kNN',experiment_info1,'R')

    score_df_knn_2 = createScoreTable('kNN',experiment_info2,'A')
    f1_df_knn_2 = createScoreTable('kNN',experiment_info2,'F')
    prec_df_knn_2 = createScoreTable('kNN',experiment_info2,'P')
    recall_df_knn_2 = createScoreTable('kNN',experiment_info2,'R')

    score_df_knn_3 = createScoreTable('kNN',experiment_info3,'A')
    f1_df_knn_3 = createScoreTable('kNN',experiment_info3,'F')
    prec_df_knn_3 = createScoreTable('kNN',experiment_info3,'P')
    recall_df_knn_3 = createScoreTable('kNN',experiment_info3,'R')


    score_df_rf_1 = createScoreTable('RF',experiment_info1,'A')
    f1_df_rf_1 = createScoreTable('RF',experiment_info1,'F')
    prec_df_rf_1 = createScoreTable('RF',experiment_info1,'P')
    recall_df_rf_1 = createScoreTable('RF',experiment_info1,'R')

    score_df_rf_2 = createScoreTable('RF',experiment_info2,'A')
    f1_df_rf_2 = createScoreTable('RF',experiment_info2,'F')
    prec_df_rf_2 = createScoreTable('RF',experiment_info2,'P')
    recall_df_rf_2 = createScoreTable('RF',experiment_info2,'R')

    score_df_rf_3 = createScoreTable('RF',experiment_info3,'A')
    f1_df_rf_3 = createScoreTable('RF',experiment_info3,'F')
    prec_df_rf_3 = createScoreTable('RF',experiment_info3,'P')
    recall_df_rf_3 = createScoreTable('RF',experiment_info3,'R')



    score_df_gbm_1 = createScoreTable('GBM',experiment_info1,'A')
    f1_df_gbm_1 = createScoreTable('GBM',experiment_info1,'F')
    prec_df_gbm_1 = createScoreTable('GBM',experiment_info1,'P')
    recall_df_gbm_1 = createScoreTable('GBM',experiment_info1,'R')

    score_df_gbm_2 = createScoreTable('GBM',experiment_info2,'A')
    f1_df_gbm_2 = createScoreTable('GBM',experiment_info2,'F')
    prec_df_gbm_2 = createScoreTable('GBM',experiment_info2,'P')
    recall_df_gbm_2 = createScoreTable('GBM',experiment_info2,'R')

    score_df_gbm_3 = createScoreTable('GBM',experiment_info3,'A')
    f1_df_gbm_3 = createScoreTable('GBM',experiment_info3,'F')
    prec_df_gbm_3 = createScoreTable('GBM',experiment_info3,'P')
    recall_df_gbm_3 = createScoreTable('GBM',experiment_info3,'R')


    # For each feature set:
    for j in range(len(featsList)):
        print(f'{j}:')
        writer1.write(f'\n{j}: ')
        writer2.write(f'\n{j}:')
        writer3.write(f'\n{j}:')

        features = featsList[j]

        # Featurized Train and Test Data
        df_train = processToFeaturized(df_train_raw,TRAINpath,FASTorAlyx=train_ForA)
        df_test1 = processToFeaturized(df_test1_raw,TEST1path,FASTorAlyx=test1_ForA) 
        df_test2 = processToFeaturized(df_test2_raw,TEST2path,FASTorAlyx=test2_ForA) 
        df_test3 = processToFeaturized(df_test3_raw,TEST3path,FASTorAlyx=test3_ForA) 
        
        # kNN classifier is covered below (train each time - not slow and normalization requires each time train)
        # Make RF classifier
        clf_rf = classifierTrain('RF',df_train)

        # Make GBM classifier
        clf_gbm = classifierTrain('GBM',df_train)


        # kNN predictions
        pred_knn_1 = classifierExperiment('kNN',df_train,df_test1)
        pred_knn_2 = classifierExperiment('kNN',df_train,df_test2)
        pred_knn_3 = classifierExperiment('kNN',df_train,df_test3)

        # RF predictions
        pred_rf_1 = makePrediction(clf_rf, df_test1)
        pred_rf_2 = makePrediction(clf_rf, df_test2)
        pred_rf_3 = makePrediction(clf_rf, df_test3)

        # GBM predictions 
        pred_gbm_1 = makePrediction(clf_gbm, df_test1)
        pred_gbm_2 = makePrediction(clf_gbm, df_test2)
        pred_gbm_3 = makePrediction(clf_gbm, df_test3)


        # Get all scores
        accuracy_score_knn_1, f1_knn_1, prec_knn_1, recall_knn_1, pred_df_knn_1 = evaluatePrediction(pred_knn_1)
        print(f'kNN 1: {accuracy_score_knn_1*100}% | {f1_knn_1} F1-Score | {prec_knn_1} Precision | {recall_knn_1} Recall | at {datetime.datetime.now()}')
        writer1.write(f'{accuracy_score_knn_1} Accuracy | {f1_knn_1} F1-Score | {prec_knn_1} Precision | {recall_knn_1} Recall | kNN | {features}\n')

        accuracy_score_knn_2, f1_knn_2, prec_knn_2, recall_knn_2, pred_df_knn_2 = evaluatePrediction(pred_knn_2)
        print(f'kNN 2: {accuracy_score_knn_2*100}% | {f1_knn_2} F1-Score | {prec_knn_2} Precision | {recall_knn_2} Recall | at {datetime.datetime.now()}')
        writer2.write(f'{accuracy_score_knn_2} Accuracy | {f1_knn_2} F1-Score | {prec_knn_2} Precision | {recall_knn_2} Recall | kNN | {features}\n')

        accuracy_score_knn_3, f1_knn_3, prec_knn_3, recall_knn_3, pred_df_knn_3 = evaluatePrediction(pred_knn_3)
        print(f'kNN 3: {accuracy_score_knn_3*100}% | {f1_knn_3} F1-Score | {prec_knn_3} Precision | {recall_knn_3} Recall | at {datetime.datetime.now()}')
        writer3.write(f'{accuracy_score_knn_3} Accuracy | {f1_knn_3} F1-Score | {prec_knn_3} Precision | {recall_knn_3} Recall | kNN | {features}\n')


        accuracy_score_rf_1, f1_rf_1, prec_rf_1, recall_rf_1, pred_df_rf_1 = evaluatePrediction(pred_rf_1)
        print(f'RF 1: {accuracy_score_rf_1*100}% | {f1_rf_1} F1-Score | {prec_rf_1} Precision | {recall_rf_1} Recall | at {datetime.datetime.now()}')
        writer1.write(f'{accuracy_score_rf_1} Accuracy | {f1_rf_1} F1-Score | {prec_rf_1} Precision | {recall_rf_1} Recall | RF | {features}\n')

        accuracy_score_rf_2, f1_rf_2, prec_rf_2, recall_rf_2, pred_df_rf_2 = evaluatePrediction(pred_rf_2)
        print(f'RF 2: {accuracy_score_rf_2*100}% | {f1_rf_2} F1-Score | {prec_rf_2} Precision | {recall_rf_2} Recall | at {datetime.datetime.now()}')
        writer2.write(f'{accuracy_score_rf_2} Accuracy | {f1_rf_2} F1-Score | {prec_rf_2} Precision | {recall_rf_2} Recall | RF | {features}\n')

        accuracy_score_rf_3, f1_rf_3, prec_rf_3, recall_rf_3, pred_df_rf_3 = evaluatePrediction(pred_rf_3)
        print(f'RF 3: {accuracy_score_rf_3*100}% | {f1_rf_3} F1-Score | {prec_rf_3} Precision | {recall_rf_3} Recall | at {datetime.datetime.now()}')
        writer3.write(f'{accuracy_score_rf_3} Accuracy | {f1_rf_3} F1-Score | {prec_rf_3} Precision | {recall_rf_3} Recall | RF | {features}\n')



        accuracy_score_gbm_1, f1_gbm_1, prec_gbm_1, recall_gbm_1, pred_df_gbm_1 = evaluatePrediction(pred_gbm_1)
        print(f'GBM 1: {accuracy_score_gbm_1*100}% | {f1_gbm_1} F1-Score | {prec_gbm_1} Precision | {recall_gbm_1} Recall | at {datetime.datetime.now()}')
        writer1.write(f'{accuracy_score_gbm_1} Accuracy | {f1_gbm_1} F1-Score | {prec_gbm_1} Precision | {recall_gbm_1} Recall | GBM | {features}\n')

        accuracy_score_gbm_2, f1_gbm_2, prec_gbm_2, recall_gbm_2, pred_df_gbm_2 = evaluatePrediction(pred_gbm_2)
        print(f'GBM 2: {accuracy_score_gbm_2*100}% | {f1_gbm_2} F1-Score | {prec_gbm_2} Precision | {recall_gbm_2} Recall | at {datetime.datetime.now()}')
        writer2.write(f'{accuracy_score_gbm_2} Accuracy | {f1_gbm_2} F1-Score | {prec_gbm_2} Precision | {recall_gbm_2} Recall | GBM | {features}\n')

        accuracy_score_gbm_3, f1_gbm_3, prec_gbm_3, recall_gbm_3, pred_df_gbm_3 = evaluatePrediction(pred_gbm_3)
        print(f'GBM 3: {accuracy_score_gbm_3*100}% | {f1_gbm_3} F1-Score | {prec_gbm_3} Precision | {recall_gbm_3} Recall | at {datetime.datetime.now()}')
        writer3.write(f'{accuracy_score_gbm_3} Accuracy | {f1_gbm_3} F1-Score | {prec_gbm_3} Precision | {recall_gbm_3} Recall | GBM | {features}\n')


        # Insert scores to tables
        # KNN 1
        score49ToTable(accuracy_score_knn_1*100,int(j),score_df_knn_1)
        score49ToTable(f1_knn_1,int(j),f1_df_knn_1)
        score49ToTable(prec_knn_1,int(j),prec_df_knn_1)
        score49ToTable(recall_knn_1,int(j),recall_df_knn_1)

        # KNN 2
        score49ToTable(accuracy_score_knn_2*100,int(j),score_df_knn_2)
        score49ToTable(f1_knn_2,int(j),f1_df_knn_2)
        score49ToTable(prec_knn_2,int(j),prec_df_knn_2)
        score49ToTable(recall_knn_2,int(j),recall_df_knn_2)

        # KNN 3
        score49ToTable(accuracy_score_knn_3*100,int(j),score_df_knn_3)
        score49ToTable(f1_knn_3,int(j),f1_df_knn_3)
        score49ToTable(prec_knn_3,int(j),prec_df_knn_3)
        score49ToTable(recall_knn_3,int(j),recall_df_knn_3)


        # RF 1
        score49ToTable(accuracy_score_rf_1*100,int(j),score_df_rf_1)
        score49ToTable(f1_rf_1,int(j),f1_df_rf_1)
        score49ToTable(prec_rf_1,int(j),prec_df_rf_1)
        score49ToTable(recall_rf_1,int(j),recall_df_rf_1)

        # RF 2
        score49ToTable(accuracy_score_rf_2*100,int(j),score_df_rf_2)
        score49ToTable(f1_rf_2,int(j),f1_df_rf_2)
        score49ToTable(prec_rf_2,int(j),prec_df_rf_2)
        score49ToTable(recall_rf_2,int(j),recall_df_rf_2)

        # RF 3
        score49ToTable(accuracy_score_rf_3*100,int(j),score_df_rf_3)
        score49ToTable(f1_rf_3,int(j),f1_df_rf_3)
        score49ToTable(prec_rf_3,int(j),prec_df_rf_3)
        score49ToTable(recall_rf_3,int(j),recall_df_rf_3)


        # GBM 1
        score49ToTable(accuracy_score_gbm_1*100,int(j),score_df_gbm_1)
        score49ToTable(f1_gbm_1,int(j),f1_df_gbm_1)
        score49ToTable(prec_gbm_1,int(j),prec_df_gbm_1)
        score49ToTable(recall_gbm_1,int(j),recall_df_gbm_1)

        # GBM 2
        score49ToTable(accuracy_score_gbm_2*100,int(j),score_df_gbm_2)
        score49ToTable(f1_gbm_2,int(j),f1_df_gbm_2)
        score49ToTable(prec_gbm_2,int(j),prec_df_gbm_2)
        score49ToTable(recall_gbm_2,int(j),recall_df_gbm_2)

        # GBM 3
        score49ToTable(accuracy_score_gbm_3*100,int(j),score_df_gbm_3)
        score49ToTable(f1_gbm_3,int(j),f1_df_gbm_3)
        score49ToTable(prec_gbm_3,int(j),prec_df_gbm_3)
        score49ToTable(recall_gbm_3,int(j),recall_df_gbm_3)


        # Save Prediction df
        savePredictionDf(pred_df_knn_1,'kNN',int(j),experiment_info1,folder1_name)
        savePredictionDf(pred_df_knn_2,'kNN',int(j),experiment_info2,folder2_name)
        savePredictionDf(pred_df_knn_3,'kNN',int(j),experiment_info3,folder3_name)

        savePredictionDf(pred_df_rf_1,'RF',int(j),experiment_info1,folder1_name)
        savePredictionDf(pred_df_rf_2,'RF',int(j),experiment_info2,folder2_name)
        savePredictionDf(pred_df_rf_3,'RF',int(j),experiment_info3,folder3_name)

        savePredictionDf(pred_df_gbm_1,'GBM',int(j),experiment_info1,folder1_name)
        savePredictionDf(pred_df_gbm_2,'GBM',int(j),experiment_info2,folder2_name)
        savePredictionDf(pred_df_gbm_3,'GBM',int(j),experiment_info3,folder3_name)


    score_df_knn_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}\\A_{score_name_knn_1}.csv")
    f1_df_knn_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}\\F_{score_name_knn_1}.csv")
    prec_df_knn_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}\\P_{score_name_knn_1}.csv")
    recall_df_knn_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_knn_1}\\R_{score_name_knn_1}.csv")

    score_df_knn_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}\\A_{score_name_knn_2}.csv")
    f1_df_knn_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}\\F_{score_name_knn_2}.csv")
    prec_df_knn_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}\\P_{score_name_knn_2}.csv")
    recall_df_knn_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_knn_2}\\R_{score_name_knn_2}.csv")

    score_df_knn_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}\\A_{score_name_knn_3}.csv")
    f1_df_knn_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}\\F_{score_name_knn_3}.csv")
    prec_df_knn_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}\\P_{score_name_knn_3}.csv")
    recall_df_knn_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_knn_3}\\R_{score_name_knn_3}.csv")


    score_df_rf_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}\\A_{score_name_rf_1}.csv")
    f1_df_rf_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}\\F_{score_name_rf_1}.csv")
    prec_df_rf_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}\\P_{score_name_rf_1}.csv")
    recall_df_rf_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_rf_1}\\R_{score_name_rf_1}.csv")

    score_df_rf_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}\\A_{score_name_rf_2}.csv")
    f1_df_rf_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}\\F_{score_name_rf_2}.csv")
    prec_df_rf_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}\\P_{score_name_rf_2}.csv")
    recall_df_rf_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_rf_2}\\R_{score_name_rf_2}.csv")

    score_df_rf_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}\\A_{score_name_rf_3}.csv")
    f1_df_rf_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}\\F_{score_name_rf_3}.csv")
    prec_df_rf_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}\\P_{score_name_rf_3}.csv")
    recall_df_rf_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_rf_3}\\R_{score_name_rf_3}.csv")


    score_df_gbm_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}\\A_{score_name_gbm_1}.csv")
    f1_df_gbm_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}\\F_{score_name_gbm_1}.csv")
    prec_df_gbm_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}\\P_{score_name_gbm_1}.csv")
    recall_df_gbm_1.to_csv(f"{documents_filepath}\\{folder1_name}\\{score_name_gbm_1}\\R_{score_name_gbm_1}.csv")

    score_df_gbm_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}\\A_{score_name_gbm_2}.csv")
    f1_df_gbm_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}\\F_{score_name_gbm_2}.csv")
    prec_df_gbm_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}\\P_{score_name_gbm_2}.csv")
    recall_df_gbm_2.to_csv(f"{documents_filepath}\\{folder2_name}\\{score_name_gbm_2}\\R_{score_name_gbm_2}.csv")

    score_df_gbm_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}\\A_{score_name_gbm_3}.csv")
    f1_df_gbm_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}\\F_{score_name_gbm_3}.csv")
    prec_df_gbm_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}\\P_{score_name_gbm_3}.csv")
    recall_df_gbm_3.to_csv(f"{documents_filepath}\\{folder3_name}\\{score_name_gbm_3}\\R_{score_name_gbm_3}.csv")


    # Close files

    writer1.close()
    writer2.close()
    writer3.close()


"""Get List of Features"""

def getFeatList():
    from itertools import product

    # 7 Spatial Representations (Pos + Euler, Pos + Quat, Pos + 6D, Pos, Euler, Quat, 6D)
    # 7 Tracker Combinations (H+D+N, H+D, H+N, D+N, H, D, N)

    position = ['_position_x','_position_y','_position_z']
    euler = ['_euler_x','_euler_y','_euler_z']
    quat = ['_quat_x','_quat_y','_quat_z','_quat_w']
    sixD = ['_sixD_a','_sixD_b','_sixD_c','_sixD_d','_sixD_e','_sixD_f']
    head = ['Head']
    lhand = ['LeftHand']
    rhand = ['RightHand']
    trackers = [[head,lhand,rhand], [head,rhand], [head,lhand], [rhand,lhand], [head], [rhand], [lhand]]
    spatial = [[position, euler], [position, quat], [position, sixD], [position], [euler], [quat], [sixD]]

    feature_list = []
    for i in range(len(trackers)):
        for j in range(len(spatial)):
            tmp = []
            for k in range(len(trackers[i])):
                for l in range(len(spatial[j])):
                    tmp = tmp + ["".join(c) for c in product(trackers[i][k],spatial[j][l])]
            feature_list.append(tmp)
    
    return feature_list


"""Processing Functions"""

def processToFeaturized(dfdict,path,FASTorAlyx):
    dfdict = dfdict.copy()
    for pid in dfdict.keys():
        current_df = dfdict[pid]
        current_df = current_df[features]
        current_df = featurize(current_df)
        dfdict[pid] = current_df

    return flattenWhole(dfdict,path,FASTorAlyx) 

def featurize(df,convert=True,samplerate='s',aggmethods=['min','max','median','mean','std']):
    # Featurize resamples by seconds, aggregates by the 5 statistics, drops NA, and flattens into one header
    
    # I added this code to convert the Timestamp column (df.index) to a datetime object
    if (convert == True):
        df.index = pd.to_datetime(df.index, format='mixed')
    
    X = df.resample(samplerate).agg(aggmethods).dropna()
    X.columns = X.columns.to_flat_index()
    return X

def flattenWhole(dfdict,path,FASTorAlyx,PIDs=None):
   # Flattens the whole dictionary of dataframes into one data frame by PID with PID number and name listed
    # I am also adding a column for Gender, since that is our response variable
    if PIDs == None:
        PIDs = dfdict.keys()
   
    if (FASTorAlyx == "FAST"):
        resplabels = getResponseLabelsFAST(path)
    elif (FASTorAlyx == "Alyx"):
        resplabels = getResponseLabelsAlyx(path)
    else:
        print("Error getting response label")
    for i,v in enumerate(PIDs):
        dfdict[v]['PIDnum'] = i
        dfdict[v]['PID'] = v
        dfdict[v][response] = resplabels[v]
    return pd.concat(dfdict.values())


"""Response Label Functions"""
def getResponseLabelsFAST(directory):
    # key = PID, value = response_label in numeric form. This will help later in flattenWhole when 
    # Creates large dataframe with labels
    resplabels = dict()
    for f in glob(path.join(directory,'*.csv')):
        pid, response_label = userIDConventionFAST(filepath=f)
        numeric_representation = convertResponseLabelToNumber(response_label)
        resplabels[pid] = numeric_representation
    return resplabels

def getResponseLabelsAlyx(directory):
    # key = PID, value = response_label in numeric form. This will help later in flattenWhole when 
    # Creates large dataframe with labels
    resplabels = dict()
    for f in glob(path.join(directory,'*.csv')):
        pid, response_label = userIDConventionAlyx(filepath=f)
        numeric_representation = convertResponseLabelToNumber(response_label)
        resplabels[pid] = numeric_representation
    return resplabels

def convertResponseLabelToNumber(response_label):
    if response_label == "M":
        retnum = 0
    elif response_label == "F":
        retnum = 1

    return retnum


"""Classifier Functions"""
def classifierExperiment(type,dftrain0,dfpred0,raw = False, verbose = 0):
    dftrain = dftrain0.copy()
    dfpred = dfpred0.copy()

    # Runs the experiment with kNN, RF, or GBM
    # Note: dftrain and dfpred will be the giant dfs of build A for one and build B for the other

    # kNN data must be normalized, so this takes care of that
    if (verbose > 0):
        print("dftrain originally:")
        print(dftrain)

    if (raw == False and type == "kNN"):
        dftrain, dfpred = normalize(dftrain,dfpred)
    elif (raw == True):
        raise Exception("Need to normalize raw, add code")

    
    if (verbose > 0):
        print("dftrain after normalize:")
        print(dftrain)

    clf=makeClassifier(type,dftrain)
    outputted_df = makePrediction(clf, dfpred)
    
    return outputted_df

def classifierTrain(type,dftrain0):
    if (type == 'kNN'):
        print('Do not call this function with kNN')
        return

    dftrain = dftrain0.copy()

    clf = makeClassifier(type,dftrain)

    return clf

def makeClassifier(cls,df):
    global classifier
    X, y = dropYColumns(df)
    
    return classifier[cls].fit(X,y)



"""Normalization Functions"""

def normalize(dftrain0,dfpred0):
    # This function will be called to normalize the train and prediction dataframes FOR KNN

    dftrain = dftrain0.copy()
    dfpred = dfpred0.copy()
    
    TX,Ty = dropYColumns(dftrain)

    Tid = dftrain[['PID','PIDnum','Gender']]
    Tcols = dftrain.drop(['PID','PIDnum','Gender'],axis=1).columns

    PX,Py = dropYColumns(dfpred)
    Pid = dfpred[['PID','PIDnum','Gender']] 
    Pcols = dfpred.drop(['PID','PIDnum','Gender'],axis=1).columns

    scaler = StandardScaler()
    scaler.fit(TX)
    
    nTX = scaler.transform(TX)
    nPX = scaler.transform(PX)
    
    Tout = pd.DataFrame(nTX,columns=Tcols,index=dftrain.index.copy())
    Pout = pd.DataFrame(nPX,columns=Pcols,index=dfpred.index.copy())

    
    Tout['PID'] = Tid['PID']
    Tout['PIDnum'] = Tid['PIDnum']
    Tout['Gender'] = Tid['Gender']

    Pout['PID'] = Pid['PID']
    Pout['PIDnum'] = Pid['PIDnum']
    Pout['Gender'] = Pid['Gender']

    return Tout,Pout

def dropYColumns(df,Xdrop=['PID','PIDnum','Gender']):
    return df.drop(Xdrop,axis=1),np.ravel(df[response])


"""Prediction Functions"""
def makePrediction(clf, df):
    df1 = df.copy()
    predX = dropYColumns(df1)[0] # Gets rid of PID, PIDnum, Gender columns. The ones we want are kept already
    
    df1['Prediction'] = clf.predict(predX)
    
    return df1

def evaluatePrediction(df,verbose = 0):
    df1 = df.copy()
    df2 = df.copy()
    df3 = df.copy()

    #Use df2 to get a list of groups
    getListOfGroups = df2.groupby('PID')
    group_list = [name for name, _ in getListOfGroups]

    if (verbose > 0):
        print(group_list)
    
    # Instead of replacing prediction column, make separate dataframe
    # Where each row is a PID, actual gender, and predicted gender, which will all be the same anyways.
    find_accuracy_df = pd.DataFrame(columns=['PID', 'Gender', 'Prediction'])
    majorityPrediction = df1.groupby('PID')['Prediction'].apply(lambda x: x.value_counts(dropna=False).idxmax()).reset_index()
    
    if (verbose > 0):
        print("Majority prediction:")
        print(majorityPrediction)

    genderPerPID = df3.groupby('PID').apply(getGender)

    if (verbose > 1):
        print("Gender per PID:")
        print(genderPerPID)

        
    #For each PID, find the index where it appears in majority prediction.
    # Use that index to get prediction for that PID.
    # Use the current PID to get row_pid
    # Use the current PID to get row_gender

    for i in range(len(group_list)):
        index_of_pid = majorityPrediction.index[majorityPrediction['PID'] == group_list[i]][0]

        row_prediction = majorityPrediction.iloc[index_of_pid]['Prediction']
        row_pid = group_list[i]
        row_gender = genderPerPID[group_list[i]]
        
        data_to_append = pd.DataFrame({'PID': [row_pid], 'Gender': [row_gender],'Prediction': [row_prediction]})
        find_accuracy_df = pd.concat([find_accuracy_df, data_to_append], ignore_index=True)

    if (verbose > 1):
        print("Find accuracy df before adding correct column")
        print(find_accuracy_df)

    # Create a correct column in this NEW dataframe
    find_accuracy_df['Correct'] = find_accuracy_df.apply(lambda x: x['Gender']==x['Prediction'],axis=1)

    if (verbose > 1):
        print("Find accuracy df before after correct column")
        print(find_accuracy_df)

    # Take the average of the Correct column to get the model's accuracy
    accuracy_score = find_accuracy_df.Correct.mean()

    true_list = (find_accuracy_df.Gender).astype(int)
    pred_list = (find_accuracy_df.Prediction).astype(int)

    f1 = f1_score(y_true=true_list, y_pred=pred_list)
    prec = precision_score(y_true=true_list, y_pred=pred_list)
    recall = recall_score(y_true=true_list, y_pred=pred_list)

    df["Accuracy Score"] = accuracy_score
    df["F1-Score"] = f1
    df["Precision"] = prec
    df["Recall"] = recall

    return accuracy_score, f1, prec, recall, df

def getGender(group):
    return int(group['Gender'].mode())


"""Functions for Saving Results"""

def savePredictionDf(pred_df,model,experimentnum,experiment_info,parentfolder):
    import os
    global documents_filepath

    cwd = os.getcwd()

    # Create an output folder
    outputdir = f"{documents_filepath}\\{parentfolder}\\{experiment_info['train_info']}-{experiment_info['test_info']}_PredDf"

    if (os.path.isdir(outputdir) == False):
        os.mkdir(outputdir)

    # Change directory to output folder
    os.chdir(outputdir)

    # Write to a file including experiment number out of 49 and model
    filename = f"{experiment_info['train_info']}-{experiment_info['test_info']}_{model}_{experimentnum}.csv"    

    # Write prediction csv to that file
    pred_df.to_csv(filename)

    # Change back to the current working directory
    os.chdir(cwd)

def createScoreTable(model,experiment_info,type):
    cols = ["P+E","P+Q","P+6D","P","E","Q","6D"]
    rows = ["H+R+L","H+R","H+L","R+L","H","R","L"]

    df = pd.DataFrame(columns=cols,index=rows)

    df.index.name = f"{type}_{experiment_info['train_info']}-{experiment_info['test_info']}_{model}"
    
    return df

def score49ToTable(score,expIndex,df):
    # Note: it is crucial that the experiment index is out of 49 as is typical

    r = int(int(expIndex)/7)
    c = int(int(expIndex)%7)

    df.iloc[r,c] = float(score)

def scoreListToExcel(score_list,df):
    # Note: this should only take a list of 49 elements.

    # i should be 0 - 48
    for i in range(len(score_list)):
        score49ToTable(score_list[i],i,df)

    return df
