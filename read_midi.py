from mido import MidiFile
import pandas as pd

import sys
import pdb

def parse_midi_messages(filename):
    mid = MidiFile(filename)

    for i, track in enumerate(mid.tracks):
        msg_list = []

        for msg in track:
            if msg.is_meta:
                continue
            elif msg.type in ['note_on', 'note_off']:
                try:
                    if msg.type == 'note_off':
                        type, velocity = 'note_on', 0
                    else:
                        type, velocity = msg.type, msg.velocity
                    msg_list.append([msg.channel, type, msg.note, msg.time, velocity])
                except Exception as e:
                    print('Error while parsing!')
                    pdb.set_trace()

        msg_df = pd.DataFrame(msg_list, columns=['channel', 'type', 'note', 'time', 'velocity'])

        if i == 4:
            pdb.set_trace()
        print('Track {}'.format(i))
        print(msg_df)


def main():
    if len(sys.argv) != 2:
        print('Usage: python read_midi.py midifilename')
        exit(0)

    filename = sys.argv[1]
    parse_midi_messages(filename)

if __name__ == '__main__':
    main()

