import sys
import os

from utils import midi_to_txt


# converts midi files into text files
# midi files are read from path given by argv[1]
# text files are saved to path given by index argv[2]
def main() -> None:
    if len(sys.argv) != 3:
        print('Usage: python mid2txt_script.py midi_directory txt_directory')
        exit(0)

    midi_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    i = 0
    for filename in os.listdir(midi_dir):
        print(i)
        i += 1
        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_to_txt(filename, input_dir=midi_dir, output_dir=txt_dir)
        else:
            pass
    

if __name__ == '__main__':
    main()