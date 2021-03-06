'''
File: model_booleanvector.py
Usage: python3 model_booleanvector.py midi_directory_path

This file defines the LSTM model for music generation.
'''

## import statements
import sys
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

# from dataset_conversion import DatasetConversion
from dataset_conversion_booleanvector import DatasetConversion
from read_midi import arr_to_midi

## class definition
class LSTMModel(object):

    def __init__(self, note_size, in_size, out_size):
        # define model
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(in_size, note_size), return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
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
            history (Keras history)

        Defines and trains the model on the given training set
        '''
        # stops training if the loss does not decrease over 3 epochs
        callback_early_stopping = EarlyStopping(monitor='loss', patience=3)
        callback_terminate_nan = TerminateOnNaN()

        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        history = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,\
                                 callbacks=[callback_early_stopping, callback_terminate_nan], verbose=1)

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

    num_examples, output_size = X.shape[0], X.shape[1]

    # transpose NOTES_SIZE x input_window_size
    X, Y = np.swapaxes(X, 1, 2), np.swapaxes(Y, 1, 2)

    # set a random example as the seed input for music generation later
    seed_idx = np.random.randint(num_examples)
    X_train, X_seed = np.delete(X, seed_idx, axis=0), X[seed_idx, :, :]
    Y_train = np.delete(Y, seed_idx, axis=0)

    note_size = 64

    generator = LSTMModel(note_size, input_window_size, output_size)
    history = generator.train_model(X_train, Y_train, batch_size=1024, epochs=30)

    # music generation!
    gen_epoch = 128
    pred_result = np.zeros((output_window_size, output_size))
    # pattern will represent the last in_size 16th notes seen
    pattern = X_seed
    out_size = 1

    for threshold in thresholds:

        for i in range(gen_epoch):
            pred = generator.predict(pattern.reshape(1, input_window_size, output_size)).reshape(output_window_size, output_size)
            bool_pred = np.array((pred > threshold).astype(int))
            # append the new output, and remove the equivalent amount of input from the start for the next prediction
            pattern = np.concatenate((pattern, bool_pred), axis=0)
            pattern = pattern[output_window_size:, :]
            pred_result = np.concatenate((pred_result, bool_pred), axis=0)

        print(pred_result)

        pred_file = arr_to_midi(pred_result[1:, :].T, \
                                filename=('real_new_output/pred_output_' + str(input_window_size) + '_' + str(step) + '_' + str(threshold) + '.mid'))

        print('The generated music is at: {}'.format(pred_file))

    return history


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 model.py midi_directory_path')
        exit(0)

    dir_path = sys.argv[1]


    for input_window_size in [16]: #, 32, 64]:
        plt.figure()
        plt.title('Losses by Step Size: Input Timestep = ' + str(input_window_size))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        for step in (np.array([.5, 1]) * input_window_size).astype(int):
        #for step in [1, 2, 4, 8]:
            thresholds = [0.04] #[0.03, 0.04, 0.05, 0.06]
            #thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
            history = test(dir_path, input_window_size, step, thresholds)
            plt.plot(history.history['loss'], label=('step=' + str(step)))
        plt.legend(loc='upper right')
        plt.savefig('real_new_output/losses_input' + str(input_window_size) + '.png')


if __name__ == '__main__':
    main()


