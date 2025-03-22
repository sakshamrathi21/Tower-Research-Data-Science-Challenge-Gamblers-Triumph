import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn import metrics as metrics
import pandas as pd
import math

def data_loader(path, table_idx, player_or_dealer):
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

def sicily_data_loader(path, player_or_dealer):
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'special_table', player_or_dealer, 'spy')]
    card = data[(f'special_table', player_or_dealer, 'card')]
    return np.array([spy, card]).T

class MyPlayerMulti:
    def __init__(self, lag=5):
        self.lag = lag
        self.player_models = {}
        self.dealer_models = {}
        self.game_history = {}
        self.table_correlations = {}
        self.prediction_history = {}
        
    def get_card_value_from_spy_value(self, value):
        """Convert spy value to card value using the given formula"""
        value += 100
        value += 2.5
        value = math.trunc(value)
        value = value % 10
        if value <= 1:
            value += 10
        return value
    
    def choose_tables(self):
        """
        Return the indices of the tables to play on.
        The Sicilian table will be appended to this list automatically.
        
        Based on preliminary analysis, tables 0 and 1 seem to have useful patterns
        that might correlate with the Sicilian table.
        """
        return [0, 1]
    
    def _analyze_correlations(self, list_spy_history_player, list_spy_history_dealer):
        """Analyze correlations between table spy values to detect patterns"""
        if any(len(hist) < self.lag for hist in list_spy_history_player):
            return
        sicily_idx = len(list_spy_history_player) - 1
        for table_idx in range(sicily_idx):
            for shift in range(1, min(len(list_spy_history_player[table_idx]), 10)):
                if len(list_spy_history_player[table_idx]) >= shift + self.lag and len(list_spy_history_player[sicily_idx]) >= self.lag:
                    shifted_player = list_spy_history_player[table_idx][-(shift+self.lag):-shift]
                    sicily_player = list_spy_history_player[sicily_idx][-self.lag:]
                    try:
                        correlation = np.corrcoef(shifted_player, sicily_player)[0, 1]
                        if abs(correlation) > 0.7:
                            key = f"player_{table_idx}_sicily"
                            self.table_correlations[key] = {
                                'shift': shift,
                                'correlation': correlation
                            }
                    except:
                        pass
                if len(list_spy_history_dealer[table_idx]) >= shift + self.lag and len(list_spy_history_dealer[sicily_idx]) >= self.lag:
                    shifted_dealer = list_spy_history_dealer[table_idx][-(shift+self.lag):-shift]
                    sicily_dealer = list_spy_history_dealer[sicily_idx][-self.lag:]
                    try:
                        correlation = np.corrcoef(shifted_dealer, sicily_dealer)[0, 1]
                        if abs(correlation) > 0.7:
                            key = f"dealer_{table_idx}_sicily"
                            self.table_correlations[key] = {
                                'shift': shift,
                                'correlation': correlation
                            }
                    except:
                        pass
    
    def _predict_next_card(self, spy_history, player_or_dealer, table_idx):
        """Predict the next card based on spy history"""
        if len(spy_history) < self.lag:
            return self.get_card_value_from_spy_value(np.mean(spy_history))
        sicily_idx = len(self.choose_tables())
        if table_idx == sicily_idx:
            # print("CHECK", self.table_correlations)
            for corr_table_idx in range(sicily_idx):
                key = f"{player_or_dealer}_{corr_table_idx}_sicily"
                if key in self.table_correlations:
                    corr_info = self.table_correlations[key]
                    shift = corr_info['shift']
                    if player_or_dealer == 'player':
                        correlated_spy_histories = self.current_spy_histories_player
                    else:
                        correlated_spy_histories = self.current_spy_histories_dealer
                    if corr_table_idx < len(correlated_spy_histories) and len(correlated_spy_histories[corr_table_idx]) > shift:
                        correlated_spy = correlated_spy_histories[corr_table_idx][-shift]
                        return self.get_card_value_from_spy_value(correlated_spy)
        recent_spy = spy_history[-self.lag:]
        if len(recent_spy) > 1:
            slope = (recent_spy[-1] - recent_spy[0]) / (len(recent_spy) - 1)
            next_spy = recent_spy[-1] + slope
        else:
            next_spy = recent_spy[-1]
        return self.get_card_value_from_spy_value(next_spy)
    
    def _make_decision(self, player_total, dealer_total, predicted_next_card, turn, table_idx):
        """Make a strategic decision based on current totals and predicted next card"""
        
        if turn == 'player':
            next_total = player_total + predicted_next_card
            if table_idx == len(self.choose_tables()):
                has_strong_correlation = any(
                    "sicily" in key and abs(info['correlation']) > 0.8 
                    for key, info in self.table_correlations.items()
                )
                print("SAKSHAM ", has_strong_correlation, next_total, predicted_next_card)
                if has_strong_correlation and next_total <= 21:
                    return "hit"
            if next_total <= 21:
                return "hit"
            return "stand"
        
        else:  
            if table_idx != len(self.choose_tables()):
                curr_dealer_total = dealer_total
                modified_dealer_total = dealer_total + predicted_next_card
                curr_player_total = player_total
                if curr_dealer_total > 16:
                    if curr_player_total >= curr_dealer_total:
                        return "continue"
                    return "surrender"
                if modified_dealer_total > 21:
                    return "continue"
                if modified_dealer_total > curr_player_total:
                    return "surrender"
                return "continue"
            else:
                has_strong_correlation = any(
                    "sicily" in key and abs(info['correlation']) > 0.8 
                    for key, info in self.table_correlations.items()
                )
                modified_dealer_total = dealer_total + predicted_next_card
                if has_strong_correlation and modified_dealer_total > 21:
                    return "continue"
                if has_strong_correlation and modified_dealer_total >= player_total:
                    return "surrender"

                return "continue"
    def get_player_action_multi(self,
                            list_curr_spy_history_player, 
                            list_curr_spy_history_dealer,
                            list_curr_card_history_player, 
                            list_curr_card_history_dealer, 
                            list_curr_player_total, 
                            list_curr_dealer_total, 
                            turn, 
                            active_tables,
                            game_index,
                            ):
        """
        Make decisions for all tables simultaneously.
        
        Returns a list of actions for each table:
        - For player's turn: "hit" or "stand"
        - For dealer's turn: "continue" or "surrender"
        """
        self.current_spy_histories_player = list_curr_spy_history_player
        self.current_spy_histories_dealer = list_curr_spy_history_dealer
        self._analyze_correlations(list_curr_spy_history_player, list_curr_spy_history_dealer)
        actions = []
        for i, active in enumerate(active_tables):
            if not active:
                actions.append("stand" if turn == "player" else "continue")
                continue
            player_spy_history = list_curr_spy_history_player[i]
            dealer_spy_history = list_curr_spy_history_dealer[i]
            player_total = list_curr_player_total[i]
            dealer_total = list_curr_dealer_total[i]
            if turn == "player":
                predicted_card = self._predict_next_card(player_spy_history, "player", i)
            else:
                predicted_card = self._predict_next_card(dealer_spy_history, "dealer", i)
            action = self._make_decision(player_total, dealer_total, predicted_card, turn, i)
            actions.append(action)
        return actions