# deep-guitarist

model.py is the main file. \
The command 'python3 model.py directory_containing_midi_files' will:\
(1) gather information from ALL MIDI files(.mid or .midi) in the directory (but not in subdirectories),\
(2) feed them through an LSTM model,\
(3) generate a midi file(pred_output.mid) containing the generated music.\
