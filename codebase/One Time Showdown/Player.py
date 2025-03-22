import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd

def data_loader(path, table_idx, player_or_dealer):
    #utility for loading train.csv, example use in the notebook
    # player_or_dealer can either by "player" or "dealer
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

class MyPlayer:
    def __init__(self,table_index):
        pass  
        
    def get_card_value_from_spy_value(self,value):
        """
        value: a value from the spy series as a float
        
        It is the same function you found in the previous part
        We will not judge this function in this part, so you can choose the return type as you prefer.
        Only make sure you return the correct value as you will be using this function
        
        The body is random  for now, rewrite accordingly

        Output:
            return a scalar value of the prediction
        """
        return 10  
        
    def get_player_spy_prediction(self,hist):
        """
        hist a 1D numpy array of size (len_history,) len_history=5
        return a scalar value of the prediction

        You should reuse the models and the code you finalised in the previous part
        Note that we will NOT judge this function here

        Output:
            return a scalar value of the prediction
        """

        return 1e6

    def get_dealer_spy_prediction(self,hist):
        """
        hist a 1D numpy array of size (len_history,) len_history=5

        You should reuse the models and the code you finalised in the previous part
        Note that we will NOT judge this function here
        
        Output:
            return a scalar value of the prediction
        """

        return 1e6

    def get_player_action(self,
                        curr_spy_history_player, 
                        curr_spy_history_dealer, 
                        curr_card_history_player, 
                        curr_card_history_dealer, 
                        curr_player_total, 
                        curr_dealer_total, 
                        turn,
                        game_index,
                        ):
        """
        Arguments:
        curr_spy_history_player: list -> real number spy value series of player observed upto this point
        curr_spy_history_dealer: list -> real number spy value series of dealer observed upto this point
        curr_card_history_player: list -> integer series of player denoting value of cards observed upto this point
        curr_card_history_dealer: list -> integer series of dealer denoting value of cards observed upto this point
        curr_player_total: integer score of player
        curr_dealer_total: integer score of dealer
        turn: string -> either "player" or "dealer" denoting if its the player drawing right now or the dealer opening her cards
        game_index: integer -> tells which game is going on. Can be useful to figure if a new game has started

        Note that correspopding series of card and spy values are of the same length

        The body is random for now, rewrite accordingly

        Output:
            if turn=="player" output either string "hit" or "stand" based on your decision
            else if turn=="dealer" output either string "surrender" or "continue" based on your decision
        """

        if turn=='player':
            return 'stand'
        else:
            return 'surrender'
