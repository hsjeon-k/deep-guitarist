import sys
import os

from utils import midi_to_txt



def main():
    if len(sys.argv) != 3:
        print('Usage: python mid2txt_script.py midi_directory txt_directory')
        exit(0)

    midi_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    i = 0
    for filename in os.listdir(midi_dir):
        print(i)
        i += 1
        if filename.endswith(".mid"):
            midi_to_txt(filename, dir=midi_dir, folder=txt_dir)
        else:
            continue
    


if __name__ == '__main__':
    main()