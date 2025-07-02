import zlib
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from features_extractor import print_midi_info
from features_extractor import replace_pitches_in_midi_file
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

def double_point_crossover(parent1, parent2):
    length = min(len(parent1), len(parent2))
    crossover_point1 = random.randint(0, length)
    crossover_point2 = random.randint(0, length)
    # Swap values if the end up being reversed
    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1
    new_parent1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
    new_parent2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
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

def evolve_single(num_generations, pop_size, ind_size, strategy=0):
    # Initialize random population
    population0 = generate_random_population(pop_size, ind_size)

    # Create guide sequences list
    guide_seq1 = midi_to_relative_pitch_sequence(midi_input_1)
    guide_seq2 = midi_to_relative_pitch_sequence(midi_input_2)
    guide_list = [guide_seq1, guide_seq2]

    # Calculate fitness value for each individual in the random population
    population1 = [(ind, calculate_fitness(ind, guide_list)) for ind in population0]

    # Track fitness values across generations for plotting
    fitness_history = []

    lower_bound = int(len(population1)*0.25) 
    lower_bound = lower_bound if lower_bound % 2 == 0 else lower_bound + 1
    upper_bound = int(len(population1)*0.75)
    upper_bound = upper_bound if upper_bound % 2 == 0 else upper_bound + 1

    for i in range(num_generations):
        print(f"Generation number: {i}")

        # Sort population by fitness in decreasing order
        sorted_pop = sorted(population1, key = lambda x: -x[1])

        # Store fitness values for current generation
        current_fitness = [ind[1] for ind in sorted_pop]
        fitness_history.append(current_fitness)

        # Remove the 25% worst individuals
        sorted_pop = sorted_pop[:upper_bound]

        # Recombine the 25% best individuals and restore population size
        for j in range(0, lower_bound, 2):
            # Recombine individuals based on chosen strategy 
            if strategy == 0:
                recombined = single_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
            elif strategy == 1:
                recombined = double_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
            elif strategy == 2:
                if i <= 200:
                    recombined = double_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                else:
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

    # Final generation fitness
    final_sorted_pop = sorted(population1, key=lambda x: -x[1])
    final_fitness = [ind[1] for ind in final_sorted_pop]
    fitness_history.append(final_fitness)
    
    return fitness_history, final_sorted_pop[0]

def evolve_multi(num_runs, plot_step_size, num_generations, pop_size, ind_size, strategy):
    all_fitness_histories = []
    best_individuals = []
    run_times = []

    total_start_time = time.time()
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        run_start_time = time.time()
        fitness_history, best_individual = evolve_single(num_generations, pop_size, ind_size, strategy)
        run_end_time = time.time()
        
        run_time = run_end_time - run_start_time
        run_times.append(run_time)

        all_fitness_histories.append(fitness_history)
        best_individuals.append(best_individual)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Aggregate fitness data across all runs for plotting
    aggregated_fitness = []
    for gen in range(num_generations):
        gen_fitness = []
        for run_fitness in all_fitness_histories:
            gen_fitness.extend(run_fitness[gen])
        aggregated_fitness.append(gen_fitness)
        
    # Find the best individual across all runs
    best_run_individual = max(best_individuals, key=lambda x: x[1])
    replace_pitches_in_midi_file(best_run_individual[0], midi_input_1, strategy)
    print("Created MIDI file using best individual's pitches")
    print(f"Best fit after {num_generations} generations across {num_runs} runs: {best_run_individual[1]}")

    # Print timing statistics
    print(f"\n=== Timing Statistics ===")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Average time per run: {np.mean(run_times):.2f}s ({np.mean(run_times)/60:.2f}min)")
    print(f"Fastest run: {min(run_times):.2f}s ({min(run_times)/60:.2f}min)")
    print(f"Slowest run: {max(run_times):.2f}s ({max(run_times)/60:.2f}min)")
    print(f"Standard deviation: {np.std(run_times):.2f}s")

    if plot_step_size == 0:
        return aggregated_fitness
    else:
        plot_fitness_evolution(aggregated_fitness, plot_step_size)

def plot_fitness_evolution(fitness_history, step_size=1):
    # Filter fitness history based on step size
    filtered_fitness = fitness_history[::step_size]
    filtered_generations = list(range(0, len(fitness_history), step_size))

    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    box_plot = plt.boxplot(filtered_fitness, patch_artist=True)
    
    # Customize the boxplot appearance
    colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_fitness)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize the plot
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    if step_size == 1:
        plt.title('Evolution of Fitness Values Across Generations (Multiple Runs)', fontsize=14, fontweight='bold')
    else:
        plt.title(f'Evolution of Fitness Values (Every {step_size} Generations, Multiple Runs)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to show generation numbers
    generation_labels = [f'Gen {i}' for i in filtered_generations]
    plt.xticks(range(1, len(generation_labels) + 1), generation_labels, rotation=45)

    # Add statistics annotations
    best_fitness_per_gen = [max(gen_fitness) for gen_fitness in filtered_fitness]
    mean_fitness_per_gen = [np.mean(gen_fitness) for gen_fitness in filtered_fitness]
    
    # Plot trend lines
    generations_x = range(1, len(filtered_fitness) + 1)
    plt.plot(generations_x, best_fitness_per_gen, 'r-', linewidth=2, alpha=0.8, label='Best Fitness')
    plt.plot(generations_x, mean_fitness_per_gen, 'b--', linewidth=2, alpha=0.8, label='Mean Fitness')
    
    # Print summary statistics
    print("\n=== Fitness Evolution Summary ===")
    print(f"Initial best fitness: {best_fitness_per_gen[0]:.4f}")
    print(f"Final best fitness: {best_fitness_per_gen[-1]:.4f}")
    print(f"Improvement: {best_fitness_per_gen[-1] - best_fitness_per_gen[0]:.4f}")
    print(f"Initial mean fitness: {mean_fitness_per_gen[0]:.4f}")
    print(f"Final mean fitness: {mean_fitness_per_gen[-1]:.4f}")

    plt.legend()
    plt.tight_layout()
    plt.show()

def multi_strategy_test(num_runs, plot_step_size, num_generations, pop_size, ind_size):
    aggregated_fitnesses_per_strategy = []
    for i in range(3):
        aggregated_fitness = evolve_multi(num_runs, plot_step_size, num_generations, pop_size, ind_size, i)
        aggregated_fitnesses_per_strategy.append(aggregated_fitness)
    
    best_fitnesses_per_strategy = []
    for strategy in aggregated_fitnesses_per_strategy:
        best_fitnesses = [max(strategy[j]) for j in range(len(strategy))]
        best_fitnesses_per_strategy.append(best_fitnesses)
    
    # Setup plotting
    plt.figure(figsize=(12,8))
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title('Evolution of Fitness Values Across Generations (Multiple Runs)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    generations_x = range(1, len(aggregated_fitnesses_per_strategy[0]) + 1)
    k = 1
    for strategy_fitnesses in best_fitnesses_per_strategy: 
        plt.plot(generations_x, strategy_fitnesses, linewidth=2, alpha=0.8, label=f"Best Fitness Strategy {k}")
        k += 1

    plt.legend()
    plt.tight_layout()
    plt.show()

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

# evolve_multi(30, 10, 1000, 500, 50)

multi_strategy_test(30, 0, 1000, 500, 50)