import zlib
import random
import numpy as np
from features_extractor import midi_to_relative_pitch_sequence

midi_input_1 = "../midi_files/mel1.mid"
midi_input_2 = "../midi_files/mel2.mid"

def mutate(relative_pitch_sequence):
    index = random.randint(0, len(relative_pitch_sequence)-1)
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
        individual = [random.randint(-20, 20) for _ in range(ind_length)]
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

def calculate_fitness(ind_seq, guide_seq):
    ncd1 = calculate_ncd(ind_seq, guide_seq[0])
    ncd2 = calculate_ncd(ind_seq, guide_seq[1])
    return 1 - ((0.5 * ncd1) + (0.5 * ncd2))

def evolve(num_generations):
    # Initialize random population
    population0 = generate_random_population(100, 50)

    # Create guide sequences list
    guide_seq1 = midi_to_relative_pitch_sequence(midi_input_1)
    guide_seq2 = midi_to_relative_pitch_sequence(midi_input_2)
    guide_list = [guide_seq1, guide_seq2]

    # Calculate fitness value for each individual in the random population
    population1 = [(ind, calculate_fitness(ind, guide_list)) for ind in population0]

    for i in range(num_generations):
        print(f"Generation number: {i}")

        # Sort population by fitness in decreasing order
        sorted_pop = sorted(population1, key = lambda x: -x[1])

        # Remove the 25% worst individuals
        sorted_pop = sorted_pop[:74]

        # Recombine the 25% best individuals and restore population size
        for j in range(0, 24, 2):
            recombined = single_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
            # Re-calculate fitness value for new recombined individuals
            recombined = [(ind, calculate_fitness(ind, guide_list)) for ind in recombined]
            # Add new individuals to current population
            sorted_pop.extend(recombined)
        
        # Mutate all individuals
        for k in range(len(sorted_pop)):
            mutate(sorted_pop[k][0])
        
        # Re-calculate fitness value for all individuals
        sorted_pop = [(ind[0], calculate_fitness(ind[0], guide_list)) for ind in sorted_pop]

        # Save new generated population for next iteration
        population1 = sorted_pop


    print(f"Best fit after {num_generations} generations: {population1[0][1]}")


### Testing zone ###

# test_pop = generate_random_population(100, 50)

# print(test_pop[0])
# print(len(test_pop))
# print(len(test_pop[0]))

# seq1 = midi_to_relative_pitch_sequence("../midi_files/mel1.mid")
# seq2 = midi_to_relative_pitch_sequence("../midi_files/Overworld.mid")

# print(min(seq1), max(seq1), len(seq1))
# print(min(seq2), max(seq2), len(seq2))

# print(calculate_ncd(seq1, seq2))

# evolve(100)