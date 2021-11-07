import numpy as np
import sys
import os

from read_midi import parse_midi_messages, arr_to_midi

# total number of notes per timestep
NOTES_SIZE = 128

# converts numpy array of ints into list of corresponding unicode character
def num_to_char(nums: np.ndarray) -> list:
    return [chr(num) for num in nums]


# converts a string into numpy array of corresponding unicode ints
def char_to_num(chars: str) -> np.ndarray:
    return np.array([ord(ch) for ch in chars])


# converts numpy array of midi track into a single compressed string
def np_to_str(track: np.ndarray) -> str:
    row, col = track.shape
    assert row == NOTES_SIZE
    txt = ""
    for j in range(col):
        cur_notes = np.around(track[:, j]).astype(int)

        # filter out indices with nonzero velocity values from cur_notes 
        indices = np.nonzero(cur_notes)[0]
        velos = cur_notes[indices]

        # concatenate indices and velos into one array
        int_arr = np.concatenate((indices, velos))

        # convert int into string and join each string
        char_arr = num_to_char(int_arr)
        txt += "".join(char_arr)
        txt += chr(NOTES_SIZE)

    return txt[:-1]


# coverts string input into a .txt file
def str_to_file(txt: str, filename: str, folder: str=None, ind: int=-1) -> str:
    dir = filename if  folder is None else os.path.join(folder, filename)
    output_name = "{}{}.txt".format(dir, "-"+str(ind) if ind > 0 else "")
    with open(output_name, "w") as text_file:
        text_file.write(txt)
    
    return output_name


# coverts .txt file into string
def file_to_str(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


# converts string input into a numpy array of midi track
def str_to_np(txt: str) -> np.ndarray:
    str_lst = txt.split(chr(NOTES_SIZE))
    result = np.zeros((NOTES_SIZE, 1))
    for s in str_lst:
        assert len(s) % 2 == 0
        half = len(s) // 2

        # get array of note_on indices and velocities
        inds = char_to_num(s[:half])
        velos = char_to_num(s[half:])

        # create current column of all notes (on and off) and append it to result array
        cur_col = np.zeros(NOTES_SIZE)
        if inds.shape[0] > 0:
            cur_col[inds] = velos
        cur_col = cur_col.reshape(NOTES_SIZE, 1)
        result = np.concatenate((result, cur_col), axis=1)
    
    return result[:, 1:]


# converts for each track of given midi file into a compressed .txt file
def midi_to_txt(filename: str, input_dir: str=None, output_dir: str=None):
    midi_path = os.path.join(input_dir, filename) if input_dir is not None else filename
    tracks = parse_midi_messages(midi_path)
    for i, track in enumerate(tracks):
        if track is not None:
            txt_list = np_to_str(track)
            if len(tracks) > 1:
                str_to_file(txt_list, filename, ind=i, folder=output_dir)
            else:
                str_to_file(txt_list, filename, folder=output_dir)


# converts a string into a midi_file
def str_to_midi(txt: str, filename: str=None):
    arr = str_to_np(txt)
    return arr_to_midi(arr, filename)


def main():
    if len(sys.argv) != 3:
        print('Usage: python utils.py midifilename number')
        exit(0)

    filename = sys.argv[1]
    if sys.argv[2] == '1':
        # tests midi-to-text conversion
        midi_to_txt(filename)

    elif sys.argv[2] == '2':
        # tests text-to-midi conversion
        txt = file_to_str(filename)
        arr = str_to_np(txt)
        print("result array has shape:", arr.shape)
        print("result array =\n", arr)
        arr_to_midi(arr)

    elif sys.argv[2] == '3':
        # tests midi-to-text, then text-to-midi conversion using generated text files
        tracks = parse_midi_messages(filename)
        for track in tracks:
            if track is not None:
                arr1 = track
        txt = np_to_str(arr1)
        path = str_to_file(txt, filename)
        txt = file_to_str(path)
        arr2 = str_to_np(txt)
        dif = arr1 - arr2
        print(np.nonzero(dif))

    else:
        print("Invalid number: use 1 or 2")
        exit(0)
    
    
if __name__ == '__main__':
    main()
