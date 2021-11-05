import numpy as np
import sys

from read_midi import parse_midi_messages 

NOTES_SIZE = 128


def num_to_char(nums) -> list:
    return [chr(num) for num in nums]


def np_to_txt(tracks):
    txt_lst = []
    for track in tracks:
        row, col = track.shape
        assert row == NOTES_SIZE
        txt = ""
        for j in range(col):
            cur_notes = np.around(track[:, j])

            # filter out indices with nonzero velocity values from cur_notes 
            indices = np.nonzero(cur_notes)
            velos = cur_notes[indices]
            cur_notes_sz = indices.shape[0]

            # build column vectors of indices and velos (velocities)
            # velos = velos.reshape(1, cur_notes_sz)
            # indices = indices.reshape(1, cur_notes_sz)

            index_velo_pairs = np.concatenate(indices, velos, axis=0)
            # assert index_velo_pairs.shape[1] == 2
            # paired_chars = num_to_char(index_velo_pairs.flatten())
            paired_chars = num_to_char(index_velo_pairs)
            txt.join(paired_chars)
            txt += " "
        txt_lst.append(txt[:-1])

    return txt_lst

def txt_to_file(txt_lst, filename):
    for i, txt in enumerate(txt_lst):
        with open("{}_{}.txt", "w".format(filename, i)) as text_file:
            text_file.write(txt)

def main():
    if len(sys.argv) != 2:
        print('Usage: python np_to_txt.py midifilename')
        exit(0)

    filename = sys.argv[1]
    tracks = parse_midi_messages(filename)
    txt_list = np_to_txt(tracks)
    txt_to_file(txt_list, filename)
    
    

if __name__ == '__main__':
    main()