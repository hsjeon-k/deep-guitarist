from mido import MidiFile
import pandas as pd
import numpy as np

import sys
import pdb

MIDI_NOTE_RANGE = 128

def parse_midi_messages(filename):
    mid = MidiFile(filename)

    for track_idx, track in enumerate(mid.tracks):
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

        notes_arr = np.zeros((MIDI_NOTE_RANGE, msg_df['time'].sum()))
        last_timestep = 0
        last_notes_on = np.zeros(MIDI_NOTE_RANGE).reshape(-1, 1)
        curr_on_set = set([])
        for msg_idx in range(len(msg_df)):
            try:
                msg_info = msg_df.iloc[msg_idx]
                note, time, velocity = msg_info.loc[['note', 'time', 'velocity']]
                if velocity == 0:
                    curr_on_set.remove(note)
                else:
                    curr_on_set.add(note)
                curr_timestep_on_notes = np.array([int(note in curr_on_set) for note in range(MIDI_NOTE_RANGE)])\
                                         .reshape(-1, 1)
                notes_arr[:, last_timestep:last_timestep + time] = last_notes_on

                last_timestep = last_timestep + time
                last_notes_on = curr_timestep_on_notes
            except Exception as e:
                pdb.set_trace()

        notes_arr[:, last_timestep:] = last_notes_on

        print('Track {}'.format(track_idx))
        print(msg_df)
        print(notes_arr)


def main():
    if len(sys.argv) != 2:
        print('Usage: python read_midi.py midifilename')
        exit(0)

    filename = sys.argv[1]
    parse_midi_messages(filename)

if __name__ == '__main__':
    main()

