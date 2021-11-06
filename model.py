import sys
import os

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

import utils
from dataset_conversion import DatasetConversion

MIDI_EXTS = ['.mid', '.midi']




class LSTM_model(object):

    def __init__(self):
        self.model = Sequential()
        self.optimizer = None

    def train_model(self, X_train, Y_train, learning_rate=0.05):
        _, in_timestep, _ = X_train.shape
        _, out_timestep, _ = Y_train.shape

        #self.model.add(LSTM(256, input_shape=(in_timestep, 1), return_sequences=True))
        self.model.add(LSTM(64, input_shape=(in_timestep, 1)))
        self.model.add(Dense(out_timestep))
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        self.model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    def predict(self, X_test):
        return self.model.predict(X_test)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 model.py midi_directory_path')
        exit(0)

    dir_path = sys.argv[1]

    dc = DatasetConversion(dir_path, sep_by_type='word')
    dc.midi_to_txt()
    X, Y = dc.txt_to_dataset(num_input=16, num_output=1)

    num_examples, _, _ = X.shape
    num_train_set = int(0.95*num_examples)
    X_train, X_test = X[:num_train_set], X[num_train_set:]
    Y_train, Y_test = Y[:num_train_set], Y[num_train_set:]

    generator = LSTM_model()
    generator.train_model(X_train, Y_train)

    Y_pred = generator.predict(X_test)

    # note that X_test is the last 5% of all EXAMPLES (possible combinations across all tracks), not all TRACKS
    pred_output = dc.dataset_to_str(Y_pred)
    true_output = dc.dataset_to_str(Y_test)

    pred_file = dc.str_to_midi(pred_output, filename='pred_output.mid')
    true_file = dc.str_to_midi(true_output, filename='true_output.mid')

    print(pred_file)
    print(true_file)

if __name__ == '__main__':
    main()

