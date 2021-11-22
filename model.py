'''
File: model.py
Usage: python3 model.py midi_directory_path

This file defines the LSTM model for music generation.
'''

## import statements
import sys
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# from dataset_conversion import DatasetConversion
from dataset_conversion_new import DatasetConversion
from read_midi import arr_to_midi

## class definition
class LSTMModel(object):

    def __init__(self, note_size, in_size, out_size):
        # define model
        self.model = Sequential()
        #self.model.add(LSTM(256, input_shape=(in_size, 1), return_sequences=True))
        #self.model.add(Dropout(0.3))
        self.model.add(LSTM(64, input_shape=(note_size, in_size), return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, input_shape=(note_size, in_size)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(out_size))
        # self.model.add(Dense(out_size, activation='sigmoid'))

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
            history (Keras history)

        Defines and trains the model on the given training set
        '''
        # stops training if the loss does not decrease over 3 epochs
        callback = EarlyStopping(monitor='loss', patience=3)

        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        history = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], verbose=1)

        return history

    def predict(self, X_seed):
        return self.model.predict(X_seed)

def test(dir_path, input_window_size, step, thresholds):
    dc = DatasetConversion(dir_path, sep_by_type='word')
    # comment out the line below if you already have MIDI files converted to text files
    # dc.midi_to_txt()

    input_window_size = input_window_size
    output_window_size = 1
    step = step
    X, Y = dc.txt_to_dataset(input_window_size=input_window_size, output_window_size=output_window_size, step=step)

    # X = X[:, 32:96, :]
    # Y = Y[:, 32:96, :]

    num_examples, output_size = X.shape[0], X.shape[1]

    # set a random example as the seed input for music generation later
    seed_idx = np.random.randint(num_examples)
    X_train, X_seed = np.delete(X, seed_idx, axis=0), X[seed_idx, :, :]
    Y_train = np.delete(Y, seed_idx, axis=0)

    note_size = 64

    generator = LSTMModel(note_size, input_window_size, output_size)
    history = generator.train_model(X_train, Y_train, batch_size=1024, epochs=30)

    # music generation!
    gen_epoch = 128
    pred_result = np.zeros((output_size, output_window_size))
    # pattern will represent the last in_size 16th notes seen
    pattern = X_seed
    out_size = 1

    for threshold in thresholds:

        for i in range(gen_epoch):
            # x = np.reshape(pattern, (1, in_size, 1))
            # predict the next out_size 16th notes from the pattern
            # pred = generator.predict(x).reshape(1, out_size, 1)
            pred = generator.predict(pattern.reshape(1, output_size, input_window_size)).reshape(output_size, output_window_size)
            bool_pred = np.array((pred > threshold).astype(int))
            # convert to string representation
            # pred_str = dc.dataset_to_str(pred)
            # append the new output, and remove the equivalent amount of input from the start for the next prediction
            pattern = np.concatenate((pattern, bool_pred), axis=1)
            pattern = pattern[:, output_window_size:]
            pred_result = np.concatenate((pred_result, bool_pred), axis=1)

        print(pred_result)

        # pred_file = dc.str_to_midi(pred_result, filename='pred_output.mid')
        pred_file = arr_to_midi(pred_result[:, 1:], \
                                filename=('pred_output_' + str(input_window_size) + '_' + str(step) + '_' + str(threshold) + '.mid'))

        print('The generated music is at: {}'.format(pred_file))

    return history


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 model.py midi_directory_path')
        exit(0)

    dir_path = sys.argv[1]

    plt.figure()
    plt.title('Losses')

    for input_window_size in [8, 16, 32, 64]:
        for step in [1, 2, 4, 8]:
            thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
            history = test(dir_path, input_window_size, step, thresholds)
            plt.plot(history['loss'], label=('input=' + str(input_window_size) + ', step=' + str(step)))

    plt.legend(loc='upper right')
    plt.savefig('losses.png')

if __name__ == '__main__':
    main()


