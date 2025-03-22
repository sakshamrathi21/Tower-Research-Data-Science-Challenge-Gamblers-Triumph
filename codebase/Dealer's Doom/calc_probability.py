import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd

def data_loader(path, table_idx, player_or_dealer):
    #utility for loading train.csv, example use in the notebook
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def bust_probability(dealer_cards):
    """
    dealer_cards: list -> integer series of player denoting value of cards observed upto this point

    Current body is random for now, change it accordingly
    
    output: probability of going bust on this table
    """
    bust_count = 0
    total_cards = len(dealer_cards)
    score = 0
    for card in dealer_cards:
        score += card
        if score > 16:
            if score > 21:
                bust_count += 1
            score = 0
    return bust_count/total_cards