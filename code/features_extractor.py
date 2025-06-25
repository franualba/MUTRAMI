import json
import pretty_midi

midi_input_1 = "../midi_files/mel1.mid"
midi_input_2 = "../midi_files/mel2.mid"

json_output = "./output.json"
midi_output = "../midi_files/output_midi.mid"

# Extract pitches from MIDI file
def midi_to_relative_pitch_sequence(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    pitch_sequence = [note.pitch for note in midi_data.instruments[0].notes]
    relative_pitch_sequence = [pitch_sequence[i+1] - pitch_sequence[i] for i in range(len(pitch_sequence)-1)]
    print(f"Number of instruments: {len(midi_data.instruments)}")
    print(f"Lower note: {min(pitch_sequence)}")
    print(f"Higher note: {max(pitch_sequence)}")
    return relative_pitch_sequence

# Extract notes from MIDI file
def midi_to_notes(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end
            })
    return notes

# Save MIDI notes to JSON file
def notes_to_json(notes, json_path):
    with open(json_path, 'w') as f:
        json.dump(notes, f, indent=4)

# Load notes from JSON file
def json_to_notes(json_path):
    with open(json_path, 'r') as f:
        notes = json.load(f)
    return notes

# Re-create MIDI file from saved JSON notes
def notes_to_midi(notes, output_path):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = 0)
    for attribute in notes:
        note = pretty_midi.Note(
            pitch = attribute['pitch'],
            velocity = attribute['velocity'],
            start = attribute['start'],
            end = attribute['end']
        )
        instrument.notes.append(note)
    midi_data.instruments.append(instrument)
    midi_data.write(output_path)


### Testing zone ###

# notes = midi_to_notes(midi_input_2)
# notes_to_json(notes, json_output)

# notes = json_to_notes(json_output)
# notes_to_midi(notes, midi_output)

# midi_to_relative_pitch_sequence(midi_input_1)

