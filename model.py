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

# from dataset_conversion import DatasetConversion
from dataset_conversion_new import DatasetConversion
from read_midi import arr_to_midi



## class definition
class LSTMModel(object):

    def __init__(self, in_size, out_size):
        # define model
        self.model = Sequential()
        #self.model.add(LSTM(256, input_shape=(in_size, 1), return_sequences=True))
        #self.model.add(Dropout(0.3))
        self.model.add(LSTM(64, input_shape=(128, in_size)))
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
        self.optimizer = Adam(learning_rate=learning_rate)
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
    # comment out the line below if you already have MIDI files converted to text files
    # dc.midi_to_txt()

    input_window_size = 16
    output_window_size = 4
    step = 20
    X, Y = dc.txt_to_dataset(input_window_size=input_window_size, output_window_size=output_window_size, step=step)

    num_examples = X.shape[0]

    # set a random example as the seed input for music generation later
    seed_idx = np.random.randint(num_examples)
    X_train, X_seed = np.delete(X, seed_idx, axis=0), X[seed_idx]
    Y_train = np.delete(Y, seed_idx, axis=0)

    generator = LSTMModel(input_window_size, output_window_size)
    generator.train_model(X_train, Y_train, batch_size=512, epochs=4)

    # music generation!
    gen_epoch = 64
    pred_result = np.zeros((128, 1))
    # pattern will represent the last in_size 16th notes seen
    pattern = X_seed
    for i in range(gen_epoch):
        # x = np.reshape(pattern, (1, in_size, 1))
        # predict the next out_size 16th notes from the pattern
        # pred = generator.predict(x).reshape(1, out_size, 1)
        pred = generator.predict(pattern).reshape(1, out_size, 1)

        # convert to string representation
        # pred_str = dc.dataset_to_str(pred)
        # append the new output, and remove the equivalent amount of input from the start for the next prediction
        pattern = np.concatenate((pattern, pred), axis=1)
        pattern = pattern[:, out_size:, :]

        pred_result = np.concatenate((pred_result, pred), axis=1)

    # pred_file = dc.str_to_midi(pred_result, filename='pred_output.mid')
    pred_file = arr_to_midi(pred_result, filename='pred_output.mid')

    print('The generated music is at: {}'.format(pred_file))

if __name__ == '__main__':
    main()

