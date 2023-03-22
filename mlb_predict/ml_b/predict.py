from . import get_data
##import get_data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Normalization
from sklearn import preprocessing
from keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

min_max_scalar = preprocessing.MinMaxScaler()
batter_fields = {'Pos': 'FLDPos', 'wRC': 'wRC+', 'OFF': 'Bat', 'fWAR': 'WAR'}
pitcher_fields = {'K': 'SO', 'K9': 'K/9', 'BB9': 'BB/9', 'HR9': 'HR/9', 'KBB': 'K/BB', 'RS9': 'RS/9', 'fWAR': 'WAR'}
fielding_fields = {'IP': 'Inn', 'DEF': 'Def', 'SB': 'BRSB', 'CS': 'BRCS'}
batting_fields = ['G', 'PA', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'wOBA', 'Bat', 'WAR', 'bWAR', 'aWAR', 'WPA']
pitching_fields = ['G', 'GS', 'IP', 'SO', 'BB', 'K/9', 'BB/9', 'HR/9', 'K/BB', 'AVG', 'WHIP', 'RS/9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WAR', 'bWAR', 'aWAR', 'WPA']

def create_model(vals):
    model = Sequential([
        Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(len(vals[0]),)),
        Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def save_predict_salaries(batters, pitchers):
    predict = pd.concat([batters[['Age', 'aWAR', 'Salary']], pitchers[['Age', 'aWAR', 'Salary']]]).dropna()
    salary = predict[['Salary']]
    predict = predict.drop(['Salary'], axis=1)
    predict_vals = predict.values
    salary_vals = salary.values
    predict_vals = predict_vals.astype(float)
    salary_vals = salary_vals.astype(float)
    predict_vals = predict_vals.view()
    salary_vals = salary_vals.view()
    salary_vals = salary_vals.reshape(-1, 1)
    predict_scaled = min_max_scalar.fit_transform(predict_vals)
    salary_scaled = min_max_scalar.fit_transform(salary_vals)
    input_train, input_val_and_test, predict_train, predict_val_and_test = train_test_split(predict_scaled, salary_scaled, test_size=0.5)
    input_val, input_test, predict_val, predict_test = train_test_split(predict_scaled, salary_scaled, test_size=0.5)
    model = Sequential([
        ##Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(len(predict_vals[0]),)),
        ##Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        ##Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        ##Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        ##Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='sigmoid', input_shape=(len(predict_vals[0]),))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    hist = model.fit(input_train, predict_train,
                batch_size=32, epochs=100,
                validation_data=(input_val, predict_val))
    model.save('./ml_b/model_weights/salaries')
    '''model.evaluate(input_test, predict_test)[1]
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()'''
    
def predict_salaries(data):
    players = pd.concat([data.get_player_batting(2022)[['Age', 'aWAR']], data.get_player_pitching(2022)[['Age', 'aWAR']]])
    player_vals = players.values
    player_vals = np.nan_to_num(player_vals)
    player_vals = player_vals.astype(float)
    model = tf.keras.models.load_model('./ml_b/model_weights/salaries')
    player_scaled = min_max_scalar.fit_transform(player_vals.view())
    predicted = model.predict(player_scaled)
    salaries = pd.concat([data.get_player_batting(2022)[['Salary']], data.get_player_pitching(2022)[['Salary']]])
    salary_vals = salaries.values
    salary_vals = np.nan_to_num(salary_vals)
    salary_vals = salary_vals.astype(float)
    min_max_scalar.fit(salary_vals.view())
    fixed_predicted = min_max_scalar.inverse_transform(predicted)
    return fixed_predicted
    
def train_pitchers(data, field: str):
    pitchers1 = data.get_player_pitching(2022)
    for i in range(2021, 2014, -1):
        pitchers1.append(data.get_player_pitching(i))
    pitchers1 = pitchers1.drop(['Dollars', 'Age Rng', 'last_name', 'first_name'], axis=1)
    pitchers = pitchers1.drop(pitchers1.columns[[x for x in range(0, pitchers1.columns.get_loc('Age'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('IFFB'), pitchers.columns.get_loc('K/9'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('IFH%'), pitchers.columns.get_loc('tERA'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('-WPA'), pitchers.columns.get_loc('O-Swing%'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('HLD'), pitchers.columns.get_loc('ERA-'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('E-F'), pitchers.columns.get_loc('Pull%'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('kwERA'), pitchers.columns.get_loc('EV'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('player_id_x'), pitchers.columns.get_loc('ba'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('ff_avg_spin'), pitchers.columns.get_loc('bWAR'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('player_id'), len(pitchers.columns))]], axis=1)
    na_rows = pd.isnull(pitchers).any(1).to_numpy().nonzero()[0].tolist()
    pitchers = pitchers.dropna()
    tot_vals = pitchers1.drop(na_rows).values
    tot_vals = np.nan_to_num(tot_vals)
    field_index = 0
    if field in pitcher_fields:
        field_index = pitchers1.columns.get_loc(pitcher_fields[field])
        pitchers = pitchers.drop([pitcher_fields[field]], axis=1)
    else:
        field_index = pitchers1.columns.get_loc(field)
        pitchers = pitchers.drop([field], axis=1)
    print(f'Training: {field}')
    vals = pitchers.values
    vals = vals.astype(float)
    pitcher_vals = vals.view()
    pitcher1_vals = tot_vals.view()
    true_vals = pitcher1_vals[:,field_index]
    true_vals = true_vals.reshape(-1, 1)
    true_vals = min_max_scalar.fit_transform(true_vals)
    pitcher_scaled = min_max_scalar.fit_transform(pitcher_vals)
    input_train, input_val_and_test, predict_train, predict_val_and_test = train_test_split(pitcher_scaled, true_vals, test_size=0.5)
    input_val, input_test, predict_val, predict_test = train_test_split(pitcher_scaled, true_vals, test_size=0.5)
    model = Sequential([
        Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(len(pitcher_vals[0]),)),
        Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    hist = model.fit(input_train, predict_train,
                batch_size=128, epochs=100,
                validation_data=(input_val, predict_val))
    return model

def train_batters(data, field: str):
    batters1 = data.get_player_batting(2022)
    for i in range(2021, 2014, -1):
        batters1.append(data.get_player_batting(i))
    batters1 = batters1.drop(['Dol', 'Age Rng'], axis=1)
    batters = batters1.drop(batters1.columns[[x for x in range(0, batters1.columns.get_loc('Age'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('GB'), batters.columns.get_loc('BB%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('Rep'), batters.columns.get_loc('WAR'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('-WPA'), batters.columns.get_loc('O-Swing%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('FA% (sc)'), batters.columns.get_loc('Pull%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('TTO%'), batters.columns.get_loc('EV'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('FLDPos'), len(batters.columns))]], axis=1)
    na_rows = pd.isnull(batters).any(1).to_numpy().nonzero()[0].tolist()
    batters = batters.dropna()
    tot_vals = batters1.drop(na_rows).values
    tot_vals = np.nan_to_num(tot_vals)
    field_index = 0
    if field in batter_fields:
        field_index = batters1.columns.get_loc(batter_fields[field])
        batters = batters.drop([batter_fields[field]], axis=1)
    else:
        field_index = batters1.columns.get_loc(field)
        batters = batters.drop([field], axis=1)
    print(f'Training: {field}')
    vals = batters.values
    vals = vals.astype(float)
    batter_vals = vals.view()
    batter1_vals = tot_vals.view()
    true_vals = batter1_vals[:,field_index]
    true_vals = true_vals.reshape(-1, 1)
    true_vals = min_max_scalar.fit_transform(true_vals)
    batter_scaled = min_max_scalar.fit_transform(batter_vals)
    input_train, input_val_and_test, predict_train, predict_val_and_test = train_test_split(batter_scaled, true_vals, test_size=0.5)
    input_val, input_test, predict_val, predict_test = train_test_split(batter_scaled, true_vals, test_size=0.5)
    model = Sequential([
        Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(len(batter_vals[0]),)),
        Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    hist = model.fit(input_train, predict_train,
                batch_size=128, epochs=100,
                validation_data=(input_val, predict_val))
    return model
    '''model.evaluate(input_test, predict_test)[1]
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()'''
    
def predict_pitcher(data, field, year):
    pitchers = data.get_player_pitching(year)
    pitchers = pitchers.drop(['Dollars', 'Age Rng', 'last_name', 'first_name'], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(0, pitchers.columns.get_loc('Age'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('IFFB'), pitchers.columns.get_loc('K/9'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('IFH%'), pitchers.columns.get_loc('tERA'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('-WPA'), pitchers.columns.get_loc('O-Swing%'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('HLD'), pitchers.columns.get_loc('ERA-'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('E-F'), pitchers.columns.get_loc('Pull%'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('kwERA'), pitchers.columns.get_loc('EV'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('player_id_x'), pitchers.columns.get_loc('ba'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('ff_avg_spin'), pitchers.columns.get_loc('bWAR'))]], axis=1)
    pitchers = pitchers.drop(pitchers.columns[[x for x in range(pitchers.columns.get_loc('player_id'), len(pitchers.columns))]], axis=1)
    pitchers = pitchers.drop([field], axis=1)
    vals = pitchers.values
    vals = np.nan_to_num(vals)
    vals = vals.astype(float)
    pitcher_vals = vals.view()
    pitcher_scaled = min_max_scalar.fit_transform(pitcher_vals)
    model = tf.keras.models.load_model(f'./ml_b/model_weights/pitcher_{field}')
    predicted = model.predict(pitcher_scaled, verbose=0)
    return predicted
    
def predict_batter(data, field, year):
    batters = data.get_player_batting(year)
    batters = batters.drop(['Dol', 'Age Rng'], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(0, batters.columns.get_loc('Age'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('GB'), batters.columns.get_loc('BB%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('Rep'), batters.columns.get_loc('WAR'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('-WPA'), batters.columns.get_loc('O-Swing%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('FA% (sc)'), batters.columns.get_loc('Pull%'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('TTO%'), batters.columns.get_loc('EV'))]], axis=1)
    batters = batters.drop(batters.columns[[x for x in range(batters.columns.get_loc('FLDPos'), len(batters.columns))]], axis=1)
    batters = batters.drop([field], axis=1)
    vals = batters.values
    vals = np.nan_to_num(vals)
    vals = vals.astype(float)
    batter_vals = vals.view()
    batter_scaled = min_max_scalar.fit_transform(batter_vals)
    model = tf.keras.models.load_model(f'./ml_b/model_weights/batter_{field}')
    predicted = model.predict(batter_scaled, verbose=0)
    return predicted
    
def create_pitcher_models(data):
    for field in pitching_fields:
        model = train_pitchers(data, field)
        print(field)
        model.save(f'./ml_b/model_weights/pitcher_{field}')
        
def create_batter_models(data):
    for field in batting_fields:
        model = train_batters(data, field)
        print(field)
        model.save(f'./ml_b/model_weights/batter_{field}')
        
    
if __name__ == '__main__':
    data = get_data.Data()
    data.get_data('', True, True, True, False, False, False, False, False)
    ##save_predict_salaries(data.get_player_batting(2022), data.get_player_pitching(2022))
    ##print(predict_salaries(data)[:10])