# Author: Harshdeep Gupta
# Date: 21 September, 2018
# Description: A file for providing utility functions
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import torch

# workspace imports
from evaluate import evaluate_model
from Dataset import MovieLensDataset

def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch_no, device, verbose = 1):
    'trains the model for one epoch and returns the loss'
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for feed_dict in data_loader:
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = feed_dict[key].to(dtype = torch.long, device = device)
        # get the predictions
        prediction = model(feed_dict)
        # print(prediction.shape)
        # get the actual targets
        rating = feed_dict['rating']
        
      
        # convert to float and change dim from [batch_size] to [batch_size,1]
        rating = rating.float().view(prediction.size())  
        loss = loss_fn(prediction, rating)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss
        



def test(model, full_dataset : MovieLensDataset, topK):
    'Test the HR and NDCG for the model @topK'
    # put the model in eval mode before testing
    if hasattr(model,'eval'):
        # print("Putting the model in eval mode")
        model.eval()
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, full_dataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Eval: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time()-t1))
    return hr, ndcg
    



def plot_statistics(hr_list, ndcg_list, loss_list, model_alias, path):
    'plots and saves the figures to a local directory'
    plt.figure()
    hr = np.array(hr_list)
    ndcg = np.array(ndcg_list)
    loss = np.array(loss_list)
    plt.plot(hr[:,0], hr[:,1],linestyle='-', marker='o', label = "HR")
    plt.plot(ndcg[:,0], ndcg[:,1],linestyle='-', marker='v', label = "NDCG")
    plt.plot(loss[:,0], loss[:,1],linestyle='-', marker='s', label = "Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(path+model_alias+".jpg")
    return




def get_items_interacted(user_id, interaction_df):
    # returns a set of items the user has interacted with
    userid_mask = interaction_df['userid'] == user_id
    interacted_items = interaction_df.loc[userid_mask].courseid
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])



def save_to_csv(df,path, header = False, index = False, sep = '\t', verbose = False):
    if verbose:
        print("Saving df to path: {}".format(path))
        print("Columns in df are: {}".format(df.columns.tolist()))

    df.to_csv(path, header = header, index = index, sep = sep)