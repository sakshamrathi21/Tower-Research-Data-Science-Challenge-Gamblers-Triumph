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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

def data_loader(path, table_idx, player_or_dealer):
    #utility for loading train.csv, example use in the notebook
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class MyPlayer:
    def __init__(self, table_index):
        self.table_index = table_index
        
        # Create empty placeholders for models
        self.player_model = None
        self.dealer_model = None
        self.player_model_type = None
        self.dealer_model_type = None
        self.player_scaler = None
        self.dealer_scaler = None
        
        # Load or train models based on table_index
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load pre-trained models or train new ones"""
        train_file = 'train.csv'  # Assuming train.csv is in the current directory
        
        # Check if we have models trained for this table
        player_model_path = f'player_model_table_{self.table_index}.pkl'
        dealer_model_path = f'dealer_model_table_{self.table_index}.pkl'
        player_scaler_path = f'player_scaler_table_{self.table_index}.pkl'
        dealer_scaler_path = f'dealer_scaler_table_{self.table_index}.pkl'
        player_type_path = f'player_type_table_{self.table_index}.txt'
        dealer_type_path = f'dealer_type_table_{self.table_index}.txt'
        
        # Try to load existing models
        if os.path.exists(player_model_path) and os.path.exists(dealer_model_path):
            # Load player model and scaler
            with open(player_model_path, 'rb') as f:
                self.player_model = pickle.load(f)
            with open(player_type_path, 'r') as f:
                self.player_model_type = f.read().strip()
            if os.path.exists(player_scaler_path):
                with open(player_scaler_path, 'rb') as f:
                    self.player_scaler = pickle.load(f)
            
            # Load dealer model and scaler
            with open(dealer_model_path, 'rb') as f:
                self.dealer_model = pickle.load(f)
            with open(dealer_type_path, 'r') as f:
                self.dealer_model_type = f.read().strip()
            if os.path.exists(dealer_scaler_path):
                with open(dealer_scaler_path, 'rb') as f:
                    self.dealer_scaler = pickle.load(f)
        else:
            # If models don't exist, train them
            if os.path.exists(train_file):
                # Train models using data from train.csv
                self._train_models(train_file)
            else:
                # If train.csv doesn't exist, use default models
                # This is a fallback for evaluation
                self.player_model_type = 'linear'
                self.dealer_model_type = 'linear'
                self.player_model = LinearRegression()
                self.dealer_model = LinearRegression()
    
    def _train_models(self, train_file):
        """Train models using data from train.csv"""
        try:
            # Load data
            data = pd.read_csv(train_file, header=[0, 1, 2])
            player_spy = data[(f'table_{self.table_index}', 'player', 'spy')].values
            dealer_spy = data[(f'table_{self.table_index}', 'dealer', 'spy')].values
            
            # Prepare training data with lag features
            X_player, y_player = self._create_sequence_data(player_spy)
            X_dealer, y_dealer = self._create_sequence_data(dealer_spy)
            
            # Analyze the data to choose the best model for each series
            player_model_info = self._select_and_train_best_model(X_player, y_player, 'player')
            dealer_model_info = self._select_and_train_best_model(X_dealer, y_dealer, 'dealer')
            
            # Store the trained models and their types
            self.player_model = player_model_info['model']
            self.player_model_type = player_model_info['type']
            self.player_scaler = player_model_info.get('scaler')
            
            self.dealer_model = dealer_model_info['model']
            self.dealer_model_type = dealer_model_info['type']
            self.dealer_scaler = dealer_model_info.get('scaler')
            
            # Save models
            self._save_models()
        except Exception as e:
            print(f"Error during training: {e}")
            # Fallback to simple linear models
            self.player_model_type = 'linear'
            self.dealer_model_type = 'linear'
            self.player_model = LinearRegression()
            self.dealer_model = LinearRegression()
    
    def _create_sequence_data(self, series, lag=5):
        """Create sequence data with lag features for time series prediction"""
        X, y = [], []
        for i in range(lag, len(series)):
            X.append(series[i-lag:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    
    def _select_and_train_best_model(self, X, y, role):
        """Select and train the best model for the data based on simple validation"""
        if len(X) < 20:  # If data is too small, use linear regression
            model = LinearRegression()
            model.fit(X, y)
            return {'model': model, 'type': 'linear'}
        
        # Split data for validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Try different models and pick the best one
        models = {
            'linear': LinearRegression(),
            'forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'nn': None  # We'll handle this separately
        }
        
        best_mse = float('inf')
        best_model_info = None
        
        # Test sklearn models first
        for model_type, model in models.items():
            if model_type == 'nn':
                continue  # Skip NN for now
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model_info = {'model': model, 'type': model_type}
        
        # Try neural network if we have enough data
        if len(X_train) > 50:
            # Scale data for NN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            
            # Create and train NN
            model = SimpleNN(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            # Train the model
            epochs = 200
            for epoch in range(epochs):
                # Forward pass
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_tensor).numpy().flatten()
            
            mse = np.mean((y_val - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model_info = {'model': model, 'type': 'nn', 'scaler': scaler}
        
        # Train the best model on all data
        if best_model_info['type'] == 'nn':
            # For NN, we need to scale and convert to tensors
            X_scaled = best_model_info['scaler'].fit_transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1))
            
            model = SimpleNN(X.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            # Train on all data
            epochs = 300
            for epoch in range(epochs):
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            best_model_info['model'] = model
        else:
            # For sklearn models, just fit on all data
            best_model_info['model'].fit(X, y)
        
        return best_model_info
    
    def _save_models(self):
        """Save trained models to disk"""
        # Save player model and type
        with open(f'player_model_table_{self.table_index}.pkl', 'wb') as f:
            pickle.dump(self.player_model, f)
        with open(f'player_type_table_{self.table_index}.txt', 'w') as f:
            f.write(self.player_model_type)
        if self.player_scaler is not None:
            with open(f'player_scaler_table_{self.table_index}.pkl', 'wb') as f:
                pickle.dump(self.player_scaler, f)
        
        # Save dealer model and type
        with open(f'dealer_model_table_{self.table_index}.pkl', 'wb') as f:
            pickle.dump(self.dealer_model, f)
        with open(f'dealer_type_table_{self.table_index}.txt', 'w') as f:
            f.write(self.dealer_model_type)
        if self.dealer_scaler is not None:
            with open(f'dealer_scaler_table_{self.table_index}.pkl', 'wb') as f:
                pickle.dump(self.dealer_scaler, f)
    
    def get_card_value_from_spy_value(self, value):
        """
        value: a value from the spy series as a float
        Output: return a scalar value of the prediction
        """
        # Create a transformed value
        value += 100
        value += 2.5
        value = math.trunc(value)
        value = value % 10
        if value <= 1:
            value += 10
        return value
    
    def get_player_spy_prediction(self, hist):
        """
        hist: a 1D numpy array of size (len_history,) len_history=5
        Output: return a scalar value of the prediction
        """
        # Reshape the history for prediction
        X = hist.reshape(1, -1)
        
        # Make prediction based on model type
        if self.player_model_type == 'nn':
            # For neural network, scale and convert to tensor
            X_scaled = self.player_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Set model to evaluation mode and make prediction
            self.player_model.eval()
            with torch.no_grad():
                prediction = self.player_model(X_tensor).item()
        else:
            # For sklearn models
            prediction = self.player_model.predict(X)[0]
        
        return prediction
    
    def get_dealer_spy_prediction(self, hist):
        """
        hist: a 1D numpy array of size (len_history,) len_history=5
        Output: return a scalar value of the prediction
        """
        # Reshape the history for prediction
        X = hist.reshape(1, -1)
        
        # Make prediction based on model type
        if self.dealer_model_type == 'nn':
            # For neural network, scale and convert to tensor
            X_scaled = self.dealer_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Set model to evaluation mode and make prediction
            self.dealer_model.eval()
            with torch.no_grad():
                prediction = self.dealer_model(X_tensor).item()
        else:
            # For sklearn models
            prediction = self.dealer_model.predict(X)[0]
        
        return prediction
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
        player_spy_prediction = 0
        if len(curr_spy_history_player) < 5:
            # we should concatenate 0 at the beginning to make it 5
            curr_spy_history_dealer = [0]*(5-len(curr_spy_history_player)) + curr_spy_history_player
            # player_spy_prediction = np.mean(curr_spy_history_player)
        player_spy_prediction = self.get_player_spy_prediction(np.array(curr_spy_history_player[-5:]))
        player_card_prediction = self.get_card_value_from_spy_value(player_spy_prediction)
        modified_player_total = curr_player_total + player_card_prediction
        dealer_spy_prediction = 0
        if len(curr_spy_history_dealer) < 5:
            curr_spy_history_dealer = [0]*(5-len(curr_spy_history_dealer)) + curr_spy_history_dealer
            # dealer_spy_prediction = np.mean(curr_spy_history_dealer)
        dealer_spy_prediction = self.get_dealer_spy_prediction(np.array(curr_spy_history_dealer[-5:]))
        dealer_card_prediction = self.get_card_value_from_spy_value(dealer_spy_prediction)
        modified_dealer_total = curr_dealer_total + dealer_card_prediction
        if turn=='player':
            if curr_dealer_total >= 20:
                return "stand"
            if curr_player_total > 16:
                return "stand"
            return "hit"
        else:
            if modified_dealer_total >= 20:
                return "surrender"
            if modified_player_total > 21:
                return "surrender"
            return "continue"
        # if turn=='player':
        #     if modified_player_total > 21:
        #         return "stand"
        #     if modified_player_total == 21:
        #         return "hit"
        #     if modified_dealer_total > 21:
        #         return "stand"
        #     if modified_dealer_total > 16:
        #         if modified_player_total > modified_dealer_total:
        #             return "stand"
        #         return "hit"
        #     return "hit"
                

            
        # else:
        #     if modified_dealer_total > 21:
        #         return "surrender"
        #     if modified_dealer_total > 16:
        #         if modified_player_total > 21:
        #             return "surrender"
        #         return "continue"
        #     return "continue"
            