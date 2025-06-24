import zlib
import random
import numpy as np
from features_extractor import midi_to_relative_pitch_sequence

midi_input_1 = "../midi_files/mel1.mid"
midi_input_2 = "../midi_files/mel2.mid"

def mutate(relative_pitch_sequence):
    index = random.randint(0, len(relative_pitch_sequence))
    offset = random.randint(-2, 2)
    relative_pitch_sequence[index] += offset
    return index

def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, min(len(parent1), len(parent2)))
    new_parent1 = parent1[:crossover_point] + parent2[crossover_point:]
    new_parent2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [new_parent1, new_parent2]

def generate_random_population(pop_size, ind_length):
    population = []
    for _ in range(pop_size):
        individual = [random.randint(-10, 10) for _ in range(ind_length)]
        population.append(individual)
    return population

def compress_seq(seq):
    bytes_seq = np.array(seq, dtype = np.int16).tobytes()
    return len(zlib.compress(bytes_seq))

def calculate_ncd(seq1, seq2):
    c1 = compress_seq(seq1)
    c2 = compress_seq(seq2)
    c12 = compress_seq(seq1 + seq2)
    return (c12 - min(c1, c2)) / max(c1, c2)

# test_pop = generate_random_population(100, 50)

# print(test_pop[0])
# print(len(test_pop))
# print(len(test_pop[0]))

seq1 = midi_to_relative_pitch_sequence("../midi_files/mel1.mid")
seq2 = midi_to_relative_pitch_sequence("../midi_files/Overworld.mid")

# print(min(seq1), max(seq1), len(seq1))
# print(min(seq2), max(seq2), len(seq2))

print(calculate_ncd(seq1, seq2))

