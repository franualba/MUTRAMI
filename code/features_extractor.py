import json
import pretty_midi
import matplotlib.pyplot as plt

midi_input_1 = "../midi_files/mel1.mid"
midi_input_2 = "../midi_files/mel2.mid"

json_output = "./output.json"
midi_output = "../midi_files/output_midi.mid"

def print_midi_info(midi_file_path, plot_histogram = False):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    pitch_sequence = [note.pitch for note in midi_data.instruments[0].notes]
    note_durations = [note.end - note.start for note in midi_data.instruments[0].notes]
    print(f"Number of instruments: {len(midi_data.instruments)}")
    print(f"Instrument program value: {midi_data.instruments[0].program}")
    print(f"Lower note: {min(pitch_sequence)}")
    print(f"Higher note: {max(pitch_sequence)}")
    print(f"Avg note duration: {round(sum(note_durations)/len(note_durations), 2)}s")
    print(f"Tempo: {round(midi_data.estimate_tempo(), 2)}bpm")
    time_signature_changes = midi_data.time_signature_changes[0]
    print(f"Time signature: {time_signature_changes.numerator}/{time_signature_changes.denominator}")
    if (plot_histogram):
        histogram = midi_data.get_pitch_class_histogram()
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.figure(figsize=(10, 6))
        plt.bar(note_names, histogram)
        plt.title('Pitch Class Histogram')
        plt.xlabel('Pitch Class')
        plt.ylabel('Relative Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Extract pitches from MIDI file
def midi_to_relative_pitch_sequence(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    pitch_sequence = [note.pitch for note in midi_data.instruments[0].notes]
    relative_pitch_sequence = [pitch_sequence[i+1] - pitch_sequence[i] for i in range(len(pitch_sequence)-1)]
    #print(relative_pitch_sequence)
    return relative_pitch_sequence

def midi_to_duration_sequence(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    durations = [note.end - note.start for note in midi_data.instruments[0].notes]
    # print(midi_data.instruments[0].notes[0].start)
    # print(midi_data.instruments[0].notes[0].end)
    # print(midi_data.instruments[0].notes[1].end - midi_data.instruments[0].notes[1].start)
    return durations

def midi_to_rhythm_sequence(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    notes = midi_data.instruments[0].notes
    rhythm_seq = []
    for i in range(len(notes) - 1):
        rhythm_seq.append(notes[i+1].start - notes[i].end)
    # Pad to match note count
    # rhythm_seq.append(0.0)
    return rhythm_seq

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

# Replace pitches in a given MIDI file with the ones obtained by the genetic algorithm 
def replace_pitches_in_midi_file(pitch_seq, midi_file_path, midi_filename):
    midi_base = pretty_midi.PrettyMIDI(midi_file_path)
    base_notes = midi_base.instruments[0].notes
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = midi_base.instruments[0].program)
    for i in range(len(pitch_seq)):
        note = pretty_midi.Note(
            pitch = pitch_seq[i] + 50,
            velocity = base_notes[i].velocity,
            start = base_notes[i].start,
            end = base_notes[i].end
        )
        instrument.notes.append(note)
    midi_output.instruments.append(instrument)
    midi_output.write(f"../midi_files/{midi_filename}.mid")

def replace_durations_in_midi_file(duration_seq, midi_file_path, midi_filename):
    midi_base = pretty_midi.PrettyMIDI(midi_file_path)
    base_notes = midi_base.instruments[0].notes
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = midi_base.instruments[0].program)
    start_time = base_notes[0].start
    for i in range(len(duration_seq)):
        duration = max(0.8, duration_seq[i])
        duration = min()
        print(duration)
        end_time = start_time + duration
        note = pretty_midi.Note(
            pitch = base_notes[i].pitch,
            velocity = base_notes[i].velocity,
            start = start_time,
            end = end_time
        )
        instrument.notes.append(note)
        start_time = end_time
    midi_output.instruments.append(instrument)
    midi_output.write(f"../midi_files/{midi_filename}.mid")

def replace_rhythms_in_midi_file(rhythm_seq, midi_file_path, midi_filename):
    midi_base = pretty_midi.PrettyMIDI(midi_file_path)
    base_notes = midi_base.instruments[0].notes
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = midi_base.instruments[0].program)
    start_time = base_notes[0].start
    for i in range(len(base_notes)):
        end_time = start_time + (base_notes[i].end - base_notes[i].start)
        note = pretty_midi.Note(
            pitch = base_notes[i].pitch,
            velocity = base_notes[i].velocity,
            start = start_time,
            end = end_time
        )
        instrument.notes.append(note)
        if i < len(rhythm_seq):
            start_time = end_time + rhythm_seq[i]
    midi_output.instruments.append(instrument)
    midi_output.write(f"../midi_files/{midi_filename}.mid")

def combine_evolved_sequences_to_midi(pitch_seq, duration_seq, rhythm_seq, midi_file_path, midi_filename):
    midi_base = pretty_midi.PrettyMIDI(midi_file_path)
    base_notes = midi_base.instruments[0].notes
    # print(len(base_notes), len(duration_seq), len(rhythm_seq))
    midi_output = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program = midi_base.instruments[0].program)
    start_time = base_notes[0].start
    for i in range(len(pitch_seq)):
        # Use evolved pitch and duration
        pitch = pitch_seq[i] + 50 if pitch_seq is not None else base_notes[i].pitch
        duration = max(0.03, duration_seq[i]) if duration_seq is not None else (base_notes[i].end - base_notes[i].start)
        end_time = start_time + duration
        note = pretty_midi.Note(
            pitch = pitch,
            velocity = base_notes[i].velocity,
            start = start_time,
            end = end_time
        )
        instrument.notes.append(note)
        # Use evolved rhythm for next note's start time
        if rhythm_seq is not None and i < len(rhythm_seq):
            start_time = end_time + rhythm_seq[i]
        else:
            start_time = end_time
    midi_output.instruments.append(instrument)
    midi_output.write(f"../midi_files/{midi_filename}.mid")

### Testing zone ###

# notes = midi_to_notes(midi_input_2)
# notes_to_json(notes, json_output)

# notes = json_to_notes(json_output)
# notes_to_midi(notes, midi_output)

# midi_to_relative_pitch_sequence(midi_input_1)
# midi_to_duration_sequence(midi_input_1)

# print_midi_info(midi_input_2, True)

