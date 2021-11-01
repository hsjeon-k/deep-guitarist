# import statements
from mido import MidiFile
import pandas as pd
import numpy as np

import sys
import pdb

# constants
MIDI_NOTE_RANGE = 128
DEFAULT_TICKS_PER_BEAT = 960
DEFAULT_TIME_SIG_NUM, DEFAULT_TIME_SIG_DENOM = 4, 4


def parse_midi_messages(filename):
    '''
    Function: parse_midi_messages

    Input
        filename(str) : name of MIDI file to read
    Output
        TBD

    Reads in the provided MIDI file and converts the note arrangements into a NumPy array,
    where the rows represent note pitches (0-127) and columns represent unit time step (MIDI ticks).
    At any timestep, pressed notes have values of 1 and others have values of 0.
    '''
    mid = MidiFile(filename)
    ticks_per_beat = mid.ticks_per_beat if mid.ticks_per_beat is not None else DEFAULT_TICKS_PER_BEAT

    for track_idx, track in enumerate(mid.tracks):
        # time signature e.g., 3/4, 4/4
        time_num, time_denom = DEFAULT_TIME_SIG_NUM, DEFAULT_TIME_SIG_DENOM
        msg_list = []

        for msg in track:
            if msg.is_meta:
                if msg.type == 'time_signature':
                    time_num, time_denom = msg.numerator, msg.denominator
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

        unit_ticks = ticks_per_beat / (4 / (time_denom/4)) # ticks per 16th note

        ### TODO: change to numpy array of velocities, and squish array into 16th-note lengths per column



        notes_arr = np.zeros((MIDI_NOTE_RANGE, msg_df['time'].sum()))
        last_timestep = 0
        last_notes_on = np.zeros(MIDI_NOTE_RANGE).reshape(-1, 1)
        curr_on_set = set([])
        for msg_idx in range(len(msg_df)):
            try:
                msg_info = msg_df.iloc[msg_idx]
                note, time, velocity = msg_info.loc[['note', 'time', 'velocity']]
                if velocity == 0:
                    if note in curr_on_set: #change
                        curr_on_set.remove(note)
                else:
                    if note not in curr_on_set: #change
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
        #pdb.set_trace()
        #print(msg_df)
        #print(notes_arr.shape)
        #for i in range(len(notes_arr)):
        #    for j in range(len(notes_arr[0])):
        #        print(int(notes_arr[i][j]), end=' ')
        #    print()



def main():
    if len(sys.argv) != 2:
        print('Usage: python read_midi.py midifilename')
        exit(0)

    filename = sys.argv[1]
    parse_midi_messages(filename)

if __name__ == '__main__':
    main()

