import pretty_midi
import sys

def split_midi_tracks(input_file):
    midi = pretty_midi.PrettyMIDI(input_file)
    for i, instrument in enumerate(midi.instruments):
        new_midi = pretty_midi.PrettyMIDI()
        new_midi.instruments.append(instrument)
        output_file = f"{input_file.rsplit('.', 1)[0]}_track_{i}.mid"
        new_midi.write(output_file)
        print(f"Track {i}: {output_file}")

if len(sys.argv) != 2:
    print("Usage: python track_splitter.py <input_file.mid>")
    sys.exit(1)

split_midi_tracks(sys.argv[1])