# Author: Harshdeep Gupta
# Date: 02 October, 2018
# Description: Implements the item popularity model for recommendations


# Workspace imports
from evaluate import evaluate_model
from utils import test, plot_statistics
from Dataset import MovieLensDataset

# Python imports
import argparse
from time import time
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description="Run ItemPop")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--num_neg_test', type=int, default=100,
                        help='Number of negative instances to pair with a positive instance while testing')
    
    return parser.parse_args()


class ItemPop():
    def __init__(self, train_interaction_matrix: sp.dok_matrix):
        """
        Simple popularity based recommender system
        """
        self.__alias__ = "Item Popularity without metadata"
        # Sum the occurences of each item to get is popularity, convert to array and 
        # lose the extra dimension
        self.item_ratings = np.array(train_interaction_matrix.sum(axis=0, dtype=int)).flatten()

    def forward(self):
        pass

    def predict(self, feeddict) -> np.array:
        # returns the prediction score for each (user,item) pair in the input
        items = feeddict['item_id']
        output_scores = [self.item_ratings[itemid] for itemid in items]
        return np.array(output_scores)

    def get_alias(self):
        return self.__alias__
       


def main():
    args = parse_args()
    path = args.path
    dataset = args.dataset
    num_negatives_test = args.num_neg_test
    print("Model arguments: %s " %(args))

    topK = 10

    # Load data

    t1 = time()
    full_dataset = MovieLensDataset(path + dataset, num_negatives_test= num_negatives_test)
    train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    model = ItemPop(train)
    test(model, full_dataset, topK)

if __name__ == "__main__":
    main()
