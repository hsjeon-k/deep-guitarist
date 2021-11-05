import numpy as np
import sys
import os

from read_midi import parse_midi_messages 

NOTES_SIZE = 128

# converts numpy array of ints into list of corresponding unicode character
def num_to_char(nums: np.ndarray) -> list:
    return [chr(num) for num in nums]


# converts list of strings into nupy array of corresponding unicode ints
def char_to_num(chars: list) -> np.ndarray:
    return np.array([ord(ch) for ch in chars])


# converts numpy array of midi track into a single compressed string
def np_to_txt(track: np.ndarray) -> str:
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

        # convert int conca into string
        char_arr = num_to_char(int_arr)
        txt += "".join(char_arr)
        txt += chr(NOTES_SIZE)

    return txt[:-1]


# coverts string input into a .txt file
def txt_to_file(txt: str, filename: str, folder: str=None, ind: int=-1) -> None:
    if folder is None:
        with open("{}{}.txt".format(filename, "-"+str(ind) if ind > 0 else ""), "w") as text_file:
            text_file.write(txt)
    else:
        dir = os.path.join(folder, filename)
        with open("{}{}.txt".format(dir, "-"+str(ind) if ind > 0 else ""), "w") as text_file:
            text_file.write(txt)


# coverts .txt file into a string
def file_to_txt(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


# converts string input into a numpy array of midi track
def txt_to_np(txt: str) -> np.ndarray:
    str_lst = txt.split(chr(NOTES_SIZE))
    result = np.zeros((NOTES_SIZE, 1))
    for s in str_lst:
        assert len(s) % 2 == 0
        half = len(s) // 2
        inds = char_to_num(s[:half])
        velos = char_to_num(s[half:])
        cur_col = np.zeros(NOTES_SIZE)
        if inds.shape[0] > 0:
            cur_col[inds] = velos
        cur_col = cur_col.reshape(NOTES_SIZE, 1)
        result = np.concatenate((result, cur_col), axis=1)
    
    return result[:, 1:]


# converts for each track of given midi file into a compressed .txt file
def midi_to_txt(filename: str, dir: str="", folder: str=None):
    tracks = parse_midi_messages(dir + "/" + filename)
    for i, track in enumerate(tracks):
        if track is not None:
            txt_list = np_to_txt(track)
            txt_to_file(txt_list, filename, ind=i, folder=folder)


def main():
    if len(sys.argv) != 3:
        print('Usage: python np_to_txt.py midifilename number')
        exit(0)

    filename = sys.argv[1]
    if sys.argv[2] == '1':
        midi_to_txt(filename)
    elif sys.argv[2] == '2':
        txt = file_to_txt(filename)
        arr = txt_to_np(txt)
        print("result array has shape:", arr.shape)
        print("result array =\n", arr)
    else:
        print("Invalid number: use 1 or 2")
        exit(0)
    
    
if __name__ == '__main__':
    main()