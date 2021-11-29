
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
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from dataset_conversion import DatasetConversion

import utils

## class definition
class LSTMModel(object):

    def __init__(self, in_size, dict_size):
        # define model
        self.model = Sequential()
        # self.model.add(LSTM(128, input_shape=(1, in_size), return_sequences=True))
        # self.model.add(Dropout(0.3))
        # self.model.add(LSTM(256, return_sequences=True))
        # self.model.add(Dropout(0.3))
        # self.model.add(LSTM(128))
        self.model.add(LSTM(128, input_shape=(1, in_size)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(dict_size, activation=softmax))

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
        callback_early_stopping = EarlyStopping(monitor='loss', patience=3)
        callback_terminate_nan = TerminateOnNaN()

        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[callback_early_stopping, callback_terminate_nan])

    def predict(self, X_seed):
        return self.model.predict(X_seed)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 model.py midi_directory_path')
        exit(0)

    dir_path = sys.argv[1]

    dc = DatasetConversion(dir_path, sep_by_type='char')
    # comment out the line below if you already have MIDI files converted to text files
    # dc.midi_to_txt()
    X, Y, dict_size = dc.txt_to_dataset(num_input=16)

    num_examples, in_size, _ = X.shape

    X = np.swapaxes(X, 1, 2)

    # set a random example as the seed input for music generation later
    seed_idx = np.random.randint(num_examples)
    X_train, X_seed = np.delete(X, seed_idx, axis=0), X[seed_idx, :, :]
    Y_train = np.delete(Y, seed_idx, axis=0)

    generator = LSTMModel(in_size, dict_size)
    generator.train_model(X_train, Y_train, batch_size=1024, epochs=50, learning_rate=0.05)

    # music generation!
    gen_epoch = 256
    pred_result = ""
    for idx in range(X_seed.shape[1]):
        pred_result += dc.int_to_data[int(X_seed[0, idx] * dict_size)]
        pred_result += chr(utils.NOTES_SIZE)
    # pattern will represent the last in_size 16th notes seen
    pattern = X_seed
    for i in range(gen_epoch):
        x = np.reshape(pattern, (1, 1, in_size))
        # predict the next out_size 16th notes from the pattern
        pred = generator.predict(x)
        print(pred)
        # pred = np.argmax(pred[0])
        # pred_softmax = np.exp(pred[0]) / np.sum(np.exp(pred[0]))
        pred_int = np.random.choice(range(len(dict_size)), p=pred[0])
        print(pred_int)
        # convert to string representation
        pred_str = dc.int_to_data[pred_int]
        # append the new output, and remove the equivalent amount of input from the start for the next prediction
        pattern = np.concatenate((x, np.int64(pred_int).reshape(1, 1, 1)), axis=2)
        pattern = pattern[:, :, 1:]

        pred_result += pred_str
        pred_result += chr(utils.NOTES_SIZE)
        print(pred_str)
        print(pred_result)

    pred_file = dc.str_to_midi(pred_result, filename='pred_output.mid')

    print('The generated music is at: {}'.format(pred_file))

if __name__ == '__main__':
    main()
