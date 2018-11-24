# Author: Harshdeep Gupta
# Date: 07 September, 2018
# Description: Splits the data into train and test using the leave the latest one out strategy 


import pandas as pd
import numpy as np
from utils import save_to_csv

INPUT_PATH = 'Data/u.data'

OUTPUT_PATH_TRAIN = 'Data/movielens.train.rating'
OUTPUT_PATH_TEST = 'Data/movielens.test.rating'
USER_FIELD = 'userID'

def get_train_test_df(transactions):
    '''
    return train and test dataframe, with leave the latest one out strategy

    Args:
        transactions: the entire df of user/item transactions
    '''

    print("Size of the entire dataset:{}".format(transactions.shape))
    transactions.sort_values(by = ['timestamp'], inplace = True)
    last_transaction_mask = transactions.duplicated(subset = {USER_FIELD}, keep = "last")
    # The last transaction mask has all the latest items of people
    # We want for the test dataset, items marked with a False
    train_df = transactions[last_transaction_mask]
    test_df = transactions[~last_transaction_mask]
    
    train_df.sort_values(by=["userID", 'timestamp'], inplace = True)
    test_df.sort_values(by=["userID", 'timestamp'], inplace = True)
    return train_df, test_df



    

def report_stats(transactions, train_df, test_df):
    whole_size = transactions.shape[0]*1.0
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]
    print("Total No. of Records = {}".format(whole_size))
    print("Train size = {}, Test size = {}".format(train_size, test_size))
    print("Train % = {}, Test % ={}".format(train_size/whole_size, test_size/whole_size))


def main():

    transactions = pd.read_csv(INPUT_PATH, sep="\t", names = ['userID', 'movieID', 'rating', 'timestamp'], engine = 'python')
    # print(transactions.head())

    # convert to implicit scenario
    transactions['rating'] = 1
    
    # make the dataset
    train_df, test_df = get_train_test_df(transactions)
    save_to_csv(train_df, OUTPUT_PATH_TRAIN, header = False,index = False, verbose = 1)
    save_to_csv(test_df, OUTPUT_PATH_TEST,header = False,index = False, verbose = 1)
    report_stats(transactions, train_df, test_df)
    return 0



if __name__ == "__main__":
    main()
