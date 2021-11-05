'''
File: read_midi.py
Usage: python3 read_midi.py path_to_midi_file

This file reads in the given MIDI file and generates a 2D NumPy array representation for each track.
The rows will represent pitch (0-127 in MIDI files), and the columns will represent timestamps.
The values of each cell will be the velocity (loudness) at which the key is being pressed at that time,
as indicated in the MIDI file, or 0 if the key is not being pressed.
'''


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


def round_to_nearest_multiple(x, multiple_of):
    '''
    Function: round_to_nearest_multiple

    Input--
        x(float) : number to round
        multiple_of : integer to round x to the multiple of
    Output--
        int : nearest multiple of multiple_of for x

    This function rounds a number to the nearest multiple of a given integer.
    Returns None if multiple_of == 0.
    '''
    if multiple_of == 0:
        return None

    floor = int(x / multiple_of)
    div_decimal = x / multiple_of - floor
    mult = floor if div_decimal < 0.5 else floor + 1

    return multiple_of * mult


def parse_midi_messages(filename):
    '''
    Function: parse_midi_messages

    Input--
        filename(str) : name of MIDI file to read
    Output--
        list[2D NumPy array] : NumPy representation of played notes, one array per track in MIDI file
                               The array is None if the track is empty (i.e., no message indicating notes_on or notes_off).

    Reads in the provided MIDI file and converts the note arrangements into a NumPy array,
    where the rows represent note pitches (0-127) and columns represent each 16th note.
    Note that any subdivisions smaller than 16th note (e.g., 32nd note) are rounded to the nearest 16th note,
        in order to reduce the dimensions of the input data..
    At any timestep, pressed notes have the velocities as their values, and others have values of 0.
    '''
    # read in the MIDI file
    mid = MidiFile(filename)

    # read in the ticks_per_beat information in the file. If not specified, use default
    # MIDI ticks are the basic unit of time in MIDI files.
    ticks_per_beat = mid.ticks_per_beat if mid.ticks_per_beat is not None else DEFAULT_TICKS_PER_BEAT

    # list of NumPy representations, to be populated later
    to_return = []


    # iterate through tracks
    for track_idx, track in enumerate(mid.tracks):
        # time signature e.g., 3/4, 4/4
        # initialize as default, but will be replaced later if specified (for each track)
        time_num, time_denom = DEFAULT_TIME_SIG_NUM, DEFAULT_TIME_SIG_DENOM

        msg_list = []

        for msg in track:
            # if a meta message
            if msg.is_meta:
                # read in time signature if specified
                if msg.type == 'time_signature':
                    time_num, time_denom = msg.numerator, msg.denominator
            # for non-meta messages, we will only consider messages relevant to playing notes.
            elif msg.type in ['note_on', 'note_off']:
                try:
                    # MIDI files have two ways of indicating the end of a key press: note_on with velocity 0, or note_off.
                    # for uniformity, we will replace all note_off messages with note_on messages of velocity 0.
                    if msg.type == 'note_off':
                        type, velocity = 'note_on', 0
                    else:
                        type, velocity = msg.type, msg.velocity
                    # append relevant message information to msg_list
                    msg_list.append([msg.channel, type, msg.note, msg.time, velocity])
                except Exception as e:
                    print('Error while parsing!')
                    print(e)
                    pdb.set_trace()

        # convert to a Pandas dataframe
        # channel: MIDI channel for the message
        # type: message type (e.g., note_on)
        # note: pitch of the played note (in MIDI files, 0-127)
        # time: time (represented in MIDI ticks), relative to the last message sent
        # velocity: how loud the note is played (in MIDI files, 0-127)
        msg_df = pd.DataFrame(msg_list, columns=['channel', 'type', 'note', 'time', 'velocity'])

        # append None as the NumPy representation if the track does not contain relevant messages
        if len(msg_df) == 0:
            to_return.append(None)
            continue


        # to reduce NumPy array dimensions, first calculate the number of MIDI ticks in a 16th note
        unit_ticks = ticks_per_beat / (4 / (time_denom/4))
        # take all the timestamps in messages
        time_vals = msg_df['time']
        # for each timestamp, calculate 'how many 16th notes have passed' since the start of the track
        # note that the timestamps are listed relative to the last message sent, so we need to first calculate the cumulative time
        # the result will then be rounded to the nearest multiple of unit_ticks, then divided by unit_ticks
        # also note that 0 is appended to the front to make the caculation of rounded_time (next line) easier
        cumul_round = [0] + [int(round_to_nearest_multiple(np.sum(time_vals[:i+1]), unit_ticks) / unit_ticks)\
                             for i in range(len(msg_df))]
        # calculate each timestamp in 16th notes, relative to the previous message sent
        rounded_time = np.array([cumul_round[i+1] - cumul_round[i] for i in range(len(msg_df))])


        # initialize the 2D NumPy array
        notes_arr = np.zeros((MIDI_NOTE_RANGE, int(np.sum(rounded_time)) + 1))
        # keep track of information in the previous step
        last_timestep = 0
        last_notes_on = np.zeros(MIDI_NOTE_RANGE).reshape(-1, 1)

        # iterate through message
        for msg_idx in range(len(msg_df)):
            try:
                # get the corresponding message information
                msg_info = msg_df.iloc[msg_idx]
                # get the timestamp in 16th notes
                time = rounded_time[msg_idx]
                # get the note(pitch) and velocity information
                note, velocity = msg_info.loc[['note', 'velocity']]
                # make a new array of notes currently being played (MIDI_NOTE_RANGE x 1)
                new_notes_on = np.copy(last_notes_on)
                new_notes_on[note, 0] = velocity
                # fill all timesteps between the current and previous timesteps with the last seen array of played notes
                # this is possible because we have saved the information from the last message,
                # meaning that there has not been any changes in the played notes in the meantime
                notes_arr[:, last_timestep:last_timestep + time] = last_notes_on

                # save the current information
                last_timestep = last_timestep + time
                last_notes_on = new_notes_on
            except Exception as e:
                print('Error while generating NumPy representation!')
                print(e)
                pdb.set_trace()

        # fill the last columns of the array
        notes_arr[:, last_timestep:] = last_notes_on

        # append the NumPy representation of current track to to_return
        to_return.append(notes_arr)


    return to_return


def main():
    if len(sys.argv) != 2:
        print('Usage: python read_midi.py midifilename')
        exit(0)

    filename = sys.argv[1]
    arr_list = parse_midi_messages(filename)
    for np_arr in arr_list:
        print(np_arr)


if __name__ == '__main__':
    main()

