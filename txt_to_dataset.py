import numpy as np

import sys
import os

import utils

SEPARATOR = chr(utils.NOTES_SIZE)
DEFAULT_NUM_INPUT = 16
DEFAULT_NUM_OUTPUT = 1


def txt_to_dataset(dir_path, num_input=None, num_output=None):
    num_input_timesteps = DEFAULT_NUM_INPUT if num_input is None else num_input
    num_output_timesteps = DEFAULT_NUM_OUTPUT if num_output is None else num_output

    txtfile_list = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            txtfile_list.append(os.path.join(dir_path, filename))

    total_timesteps_per_feed = num_input_timesteps + num_output_timesteps

    inputs = []
    outputs = []

    for file in txtfile_list:
        with open(file, 'r') as in_f:
            data = in_f.read().strip('\n')
            notes_by_timestep = data.split(SEPARATOR)

            for idx in range(len(notes_by_timestep) - total_timesteps_per_feed  + 1):
                input = notes_by_timestep[idx : idx + num_input_timesteps]
                output = notes_by_timestep[idx + num_input_timesteps : idx + total_timesteps_per_feed]

                inputs.append(input)
                outputs.append(output)

    X = np.array(inputs).reshape(len(inputs), num_input_timesteps, 1)
    Y = np.array(outputs).reshape(len(outputs), num_output_timesteps, 1)

    return X, Y


def main():
    if len(sys.argv) not in [2, 4]:
        print('Usage: python3 txt_to_dataset.py directory_path, OR python3 txt_to_dataset.py directory_path num_input num_output')
        exit()

    dir_path = sys.argv[1]
    if len(sys.argv) == 4:
        num_input, num_output = sys.argv[2], sys.argv[3]
    else:
        num_input, num_output = None, None
    X, Y = txt_to_dataset(dir_path, num_input, num_output)
    print(X)
    print(Y)


if __name__ == '__main__':
    main()

