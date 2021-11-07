'''
File: model.py
Usage: python3 model.py midi_directory_path

This file defines the LSTM model for music generation.
'''

## import statements
import sys
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from dataset_conversion import DatasetConversion


## class definition
class LSTMModel(object):

    def __init__(self, in_size, out_size):
        # define model
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(in_size, 1), return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(64, input_shape=(in_size, 1)))
        self.model.add(Dense(out_size))

        self.optimizer = None

    def train_model(self, X_train, Y_train, batch_size=32, epochs=10, learning_rate=0.05):
        '''
        Class function: train_model

        Input--
            X_train(np.ndarray) : shape (n_examples, input_size, 1); training set input
            Y_train(np.ndarray) : shape (n_examples, output_size, 1); training set output
            batch_size(int)     : batch size for training
            epochs(int)         : number of epochs to train for
            learning_rate(float): learning rate of the model
        Output--
            None

        Defines and trains the model on the given training set
        '''
        self.optimizer = Adam(learning_rate=0.05)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    def predict(self, X_seed):
        return self.model.predict(X_seed)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 model.py midi_directory_path')
        exit(0)

    dir_path = sys.argv[1]

    dc = DatasetConversion(dir_path, sep_by_type='word')
    # dc.midi_to_txt()
    X, Y = dc.txt_to_dataset(num_input=16, num_output=4)

    num_examples, in_size, _ = X.shape
    _, out_size, _ = Y.shape
    # num_train_set = int(0.95*num_examples)
    # X_train, X_test = X[:num_train_set], X[num_train_set:]
    # Y_train, Y_test = Y[:num_train_set], Y[num_train_set:]

    seed_idx = np.random.randint(num_examples)
    X_train, X_seed = np.delete(X, seed_idx, axis=0), X[seed_idx, :, :]
    Y_train = np.delete(Y, seed_idx, axis=0)

    generator = LSTMModel(in_size, out_size)
    generator.train_model(X_train, Y_train, batch_size=512, epochs=10)

    # one-pass predictor
    # Y_pred = generator.predict(X_test)

    gen_epoch = 100
    pred_result = ""
    pattern = X_seed
    for i in range(gen_epoch):
        x = np.reshape(pattern, (1, in_size, 1))
        pred = generator.predict(x).reshape(1, out_size, 1)
        pred_str = dc.dataset_to_str(pred)
        pattern = np.concatenate((x, pred), axis=1)
        pattern = pattern[:, out_size:, :]
        pred_result += pred_str

    # note that X_test is the last 5% of all EXAMPLES (possible combinations across all tracks), not all TRACKS
    # pred_output = dc.dataset_to_str(Y_pred)
    # true_output = dc.dataset_to_str(Y_test)

    pred_file = dc.str_to_midi(pred_result, filename='pred_output.mid')

    print(pred_file)

if __name__ == '__main__':
    main()

