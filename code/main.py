import random

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

