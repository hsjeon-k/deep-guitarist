import numpy as np

import sys
import os

import utils

SEPARATOR = chr(utils.NOTES_SIZE)
DEFAULT_NUM_INPUT = 16
DEFAULT_NUM_OUTPUT = 1


class Dataset_conversion(object):

    def __init__(self):
        self.combination_to_int = {}
        self.int_to_combination = {}

    @staticmethod
    def get_files_by_ext(dir_path, ext):
        file_list = []
        for filename in os.listdir(dir_path):
            if filename.endswith(ext):
                file_list.append(filename)

        return file_list


    def txt_to_dataset(self, dir_path, num_input=None, num_output=None):
        num_input_timesteps = DEFAULT_NUM_INPUT if num_input is None else num_input
        num_output_timesteps = DEFAULT_NUM_OUTPUT if num_output is None else num_output

        txtfile_list = self.get_files_by_ext(dir_path, '.txt')

        total_timesteps_per_feed = num_input_timesteps + num_output_timesteps

        inputs = []
        outputs = []

        for file in txtfile_list:
            filepath = os.path.join(dir_path, file)
            with open(filepath, 'r') as in_f:
                data = in_f.read().strip('\n')
                notes_by_timestep = data.split(SEPARATOR)

                for idx in range(len(notes_by_timestep) - total_timesteps_per_feed  + 1):
                    input = notes_by_timestep[idx : idx + num_input_timesteps]
                    output = notes_by_timestep[idx + num_input_timesteps : idx + total_timesteps_per_feed]

                    inputs.append(input)
                    outputs.append(output)

        # flatten lists and join, to compute all possible combinations (vocabulary)
        possible_combinations_X = [comb for input in inputs for comb in input]
        possible_combinations_Y = [comb for output in outputs for comb in output]
        possible_combinations = possible_combinations_X + possible_combinations_Y
        # sort so that similar combinations are nearby
        # (e.g., two combinations may both start with char equivalent to 'C')
        possible_combinations = sorted(list(set(possible_combinations)))
        # assign index to each
        self.combination_to_int = dict([(comb, i) for i, comb in enumerate(possible_combinations)])
        self.int_to_combination = dict([(i, comb) for i, comb in enumerate(possible_combinations)])
        # convert combinations to ints
        input_ints = [[self.combination_to_int[comb]/len(self.combination_to_int) for comb in input] for input in inputs]
        output_ints = [[self.combination_to_int[comb]/len(self.combination_to_int) for comb in output] for output in outputs]

        X = np.array(input_ints).reshape(len(inputs), num_input_timesteps, 1)
        Y = np.array(output_ints).reshape(len(outputs), num_output_timesteps, 1)

        return X, Y

    def dataset_to_txt(self, Y):
        Y = np.around(Y * len(self.int_to_combination))
        Y = np.minimum(np.maximum(Y, 0), len(self.int_to_combination) - 1)
        Y = Y.flatten().tolist()
        Y_str = [self.int_to_combination[Y[i]] for i in range(len(Y))]
        str_output = SEPARATOR.join(Y_str)

        return str_output


def main():
    if len(sys.argv) not in [2, 4]:
        print('Usage: python3 txt_to_dataset.py directory_path, OR python3 txt_to_dataset.py directory_path num_input num_output')
        exit()

    dir_path = sys.argv[1]
    if len(sys.argv) == 4:
        num_input, num_output = sys.argv[2], sys.argv[3]
    else:
        num_input, num_output = None, None

    dc = Dataset_conversion()
    X, Y = dc.txt_to_dataset(dir_path, num_input, num_output)
    print(X)
    print(Y)


if __name__ == '__main__':
    main()

