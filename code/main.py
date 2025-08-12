import os
import zlib
import time
import random
import pretty_midi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from features_extractor import (
    midi_to_relative_pitch_sequence,
    midi_to_duration_sequence,
    midi_to_rhythm_sequence,
    replace_pitches_in_midi_file,
    replace_durations_in_midi_file,
    replace_rhythms_in_midi_file,
    combine_evolved_sequences_to_midi
)

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
    # Swap values if they end up being reversed
    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1
    new_parent1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
    new_parent2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
    return [new_parent1, new_parent2]

def triple_point_crossover(parent1, parent2):
    length = min(len(parent1), len(parent2))
    points = sorted([random.randint(0, length) for _ in range(3)])
    new_parent1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:points[2]] + parent2[points[2]:]
    new_parent2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:points[2]] + parent1[points[2]:]
    return [new_parent1, new_parent2]

def quadruple_point_crossover(parent1, parent2):
    length = min(len(parent1), len(parent2))
    points = sorted([random.randint(0, length) for _ in range(4)])
    new_parent1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:points[2]] + parent2[points[2]:points[3]] + parent1[points[3]:]
    new_parent2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:points[2]] + parent1[points[2]:points[3]] + parent2[points[3]:]
    return [new_parent1, new_parent2]

def generate_random_population(pop_size, ind_length, mode = 'pitch'):
    population = []
    for _ in range(pop_size):
        if mode == 'pitch':
            individual = [random.randint(-20, 20) for _ in range(ind_length)]
        elif mode == 'duration':
            individual = [random.uniform(0.01, 2.2) for _ in range(ind_length)]  # durations in seconds
        elif mode == 'rhythm':
            individual = [random.uniform(0.0, 1.0) for _ in range(ind_length)]  # rhythm in seconds
        population.append(individual)
    return population

def compress_seq(seq):
    arr = np.array(seq)
    if arr.dtype.kind == 'f':  # float
        bytes_seq = arr.astype(np.float64).tobytes()
    else:
        bytes_seq = arr.astype(np.int16).tobytes()
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

def evolve_single(num_generations, pop_size, ind_size, strategy = 0, seed = None, mode = 'pitch'):
    # Set random seeds if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Initialize random population
    population0 = generate_random_population(pop_size, ind_size, mode = mode)

    # Create guide sequences list
    if mode == 'pitch':
        guide_seq1 = midi_to_relative_pitch_sequence(midi_input_1)
        guide_seq2 = midi_to_relative_pitch_sequence(midi_input_2)
    elif mode == 'duration':
        guide_seq1 = midi_to_duration_sequence(midi_input_1)
        guide_seq2 = midi_to_duration_sequence(midi_input_2)
    elif mode == 'rhythm':
        guide_seq1 = midi_to_rhythm_sequence(midi_input_1)
        guide_seq2 = midi_to_rhythm_sequence(midi_input_2)
        # print(min(guide_seq1), max(guide_seq1))
        # print(min(guide_seq2), max(guide_seq2))
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
        # print(f"Generation number: {i}")

        # Sort population by fitness in decreasing order
        sorted_pop = sorted(population1, key = lambda x: -x[1])

        # Store fitness values for current generation
        current_fitness = [ind[1] for ind in sorted_pop]
        fitness_history.append(current_fitness)

        if strategy == 6:
            sorted_pop = sorted_pop[:2]
            random_pop = generate_random_population(pop_size-2, ind_size)
            random_pop_fitness = [(ind, calculate_fitness(ind, guide_list)) for ind in random_pop]
            sorted_pop.extend(random_pop_fitness)
        else:
            # Remove the 25% worst individuals
            sorted_pop = sorted_pop[:upper_bound]

            # Recombine the 25% best individuals and restore population size
            for j in range(2, lower_bound, 2):
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
                elif strategy == 3:
                    recombined = triple_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                elif strategy == 4:
                    recombined = quadruple_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                elif strategy == 5:
                    if i <= 200:
                        recombined = double_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                    elif 200 < i <= 500:
                        recombined = triple_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                    else:
                        recombined = single_point_crossover(sorted_pop[j][0], sorted_pop[j+1][0])
                # Re-calculate fitness value for new recombined individuals
                recombined = [(ind, calculate_fitness(ind, guide_list)) for ind in recombined]
                # Add new individuals to current population
                sorted_pop.extend(recombined)
            
            # Mutate all individuals
            for k in range(2, len(sorted_pop)):
                # for _ in range(10):
                mutate(sorted_pop[k][0])
            
            # Re-calculate fitness value for all individuals
            sorted_pop = [(ind[0], calculate_fitness(ind[0], guide_list)) for ind in sorted_pop]
                       
        # Save new generated population for next iteration
        population1 = sorted_pop

    # Final generation fitness
    final_sorted_pop = sorted(population1, key = lambda x: -x[1])
    final_fitness = [ind[1] for ind in final_sorted_pop]
    fitness_history.append(final_fitness)
    
    return fitness_history, final_sorted_pop[0]

def evolve_multi(
        num_runs, plot_step_size, num_generations, pop_size, ind_size, strategy, 
        base_seed = None,
        midi_filename = None,
        timing_csv_filename = None, 
        fitness_csv_filename = None, 
        boxplot_csv_filename = None,
        mode = 'pitch'
        ):
    
    all_fitness_histories = []
    best_individuals = []
    run_times = []

    total_start_time = time.time()
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Set random seeds for repeatability
        if base_seed is not None:
            run_seed = base_seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

        run_start_time = time.time()
        fitness_history, best_individual = evolve_single(
            num_generations, pop_size, ind_size, strategy, 
            seed = run_seed if base_seed is not None else None, 
            mode = mode
            )
        run_end_time = time.time()
        
        run_time = run_end_time - run_start_time
        run_times.append(run_time)

        all_fitness_histories.append(fitness_history)
        best_individuals.append(best_individual)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Aggregate fitness data across all runs for plotting and statistics
    aggregated_fitness = []
    for gen in range(num_generations):
        gen_fitness = []
        for run_fitness in all_fitness_histories:
            gen_fitness.extend(run_fitness[gen])
        aggregated_fitness.append(gen_fitness)
        
    # Find the best individual across all runs
    best_run_individual = max(best_individuals, key = lambda x: x[1])

    if midi_filename:
        if mode == 'pitch':
            replace_pitches_in_midi_file(best_run_individual[0], midi_input_1, midi_filename)
        elif mode == 'duration':
            replace_durations_in_midi_file(best_run_individual[0], midi_input_1, midi_filename)
        elif mode == 'rhythm':
            replace_rhythms_in_midi_file(best_run_individual[0], midi_input_1, midi_filename)
        print(f"Created MIDI file: {midi_filename}")
    else:
        if mode == 'pitch':
            replace_pitches_in_midi_file(best_run_individual[0], midi_input_1, strategy)
        elif mode == 'duration':
            replace_durations_in_midi_file(best_run_individual[0], midi_input_1, strategy)
        elif mode == 'rhythm':
            replace_rhythms_in_midi_file(best_run_individual[0], midi_input_1, strategy)
            print(min(best_run_individual[0]), max(best_run_individual[0]))
        print("Created MIDI file using best individual's pitches")

    print(f"Best fit after {num_generations} generations across {num_runs} runs: {round(best_run_individual[1], 4)}")

    # Prepare current strategy timing statistics
    strategy_name = f"Strategy {strategy+1}"
    def fmt(sec):
        return f"{round(sec,1)}s ({round(sec/60,1)}min)"
    timing_col = pd.Series({
        "Total execution time": fmt(total_time),
        "Average time per run": fmt(np.mean(run_times)),
        "Fastest run": fmt(min(run_times)),
        "Slowest run": fmt(max(run_times)),
        "Standard deviation": fmt(np.std(run_times)),
        "Best fitness": round(best_run_individual[1], 4)
    }, name = strategy_name)

    # Save strategy timing statistics
    if timing_csv_filename:
        if os.path.exists(timing_csv_filename):
            df_timing = pd.read_csv(timing_csv_filename, index_col=0)
            df_timing[strategy_name] = timing_col
        else:
            df_timing = pd.DataFrame(timing_col)
        df_timing.to_csv(timing_csv_filename)
        print(f"Timing statistics appended to {timing_csv_filename}")

    # Save strategy fitness statistics per generation
    if fitness_csv_filename:
        fitness_stats = []
        for gen, gen_fitness in enumerate(aggregated_fitness):
            fitness_stats.append({
                "Generation": gen+1,
                "Mean": round(np.mean(gen_fitness), 4),
                "Std": round(np.std(gen_fitness), 4),
                "Min": round(np.min(gen_fitness), 4),
                "Max": round(np.max(gen_fitness), 4)
            })
        df_fitness = pd.DataFrame(fitness_stats)
        df_fitness.to_csv(fitness_csv_filename, index=False)
        print(f"Fitness statistics saved to {fitness_csv_filename}")

    if plot_step_size == 0:
        return aggregated_fitness
    else:
        # Save aggregated fitness values to CSV for boxplot
        if boxplot_csv_filename:
            df_box = pd.DataFrame([[round(val, 4) for val in gen] for gen in aggregated_fitness])
            df_box = df_box.transpose()
            df_box.to_csv(boxplot_csv_filename, index=False)
            print(f"Aggregated fitness values saved to {boxplot_csv_filename}")
            plot_fitness_boxplot_from_csv(boxplot_csv_filename, step_size = plot_step_size)
        else:
            plot_fitness_evolution(aggregated_fitness, plot_step_size)

def multi_strategy_test(num_runs, num_generations, pop_size, ind_size, save_csv = True, base_seed = None, mode = 'pitch'):
    aggregated_fitnesses_per_strategy = []
    # Compose experiment info for filenames
    exp_info = f"{num_runs}runs_{pop_size}pop_{ind_size}ind_{num_generations}gen"
    timing_csv_filename = f"{mode}_timing_stats_{exp_info}.csv"
    fitness_csv_filenames = []

    for i in range(7):
        strategy_name = f"strategy{i+1}"
        fitness_csv_filename = f"{mode}_fitness_stats_{strategy_name}_{exp_info}.csv"
        midi_filename = f"{mode}_best_evolution_output_{strategy_name}_{exp_info}"
        fitness_csv_filenames.append(fitness_csv_filename)
        
        # Use a different base_seed per strategy for full repeatability, or keep the same for all
        strategy_seed = (base_seed + i * 10000) if base_seed is not None else None
        aggregated_fitness = evolve_multi(
            num_runs, 0, num_generations, pop_size, ind_size, i,
            base_seed = strategy_seed,
            midi_filename = midi_filename,
            timing_csv_filename = timing_csv_filename,
            fitness_csv_filename = fitness_csv_filename,
            mode = mode
        )
        aggregated_fitnesses_per_strategy.append(aggregated_fitness)
    
    best_fitnesses_per_strategy = []
    for strategy in aggregated_fitnesses_per_strategy:
        best_fitnesses = [round(max(strategy[j]), 4) for j in range(len(strategy))]
        best_fitnesses_per_strategy.append(best_fitnesses)
    
    # Save best fitnesses per generation for all strategies in one CSV
    if save_csv:
        df = pd.DataFrame()
        for k, strategy_fitnesses in enumerate(best_fitnesses_per_strategy):
            df[f"Strategy {k+1}"] = strategy_fitnesses
        fitness_csv_filename = f"{mode}_fitness_best_all_{exp_info}.csv"
        df.to_csv(fitness_csv_filename, index=False)
        print(f"Best fitness per generation for all strategies saved to {fitness_csv_filename}")

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

def plot_multi_strategy_test_from_csv(csv_filename):
    # Load data from CSV file
    df = pd.read_csv(csv_filename)
    
    # Setup plotting
    plt.figure(figsize=(12,8))
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title('Evolution of Fitness Values Across Generations (Multiple Runs)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot trend line for each strategy (each column)
    generations_x = range(1, len(df) + 1)
    for column in df.columns:
        plt.plot(generations_x, df[column], linewidth=2, alpha=0.8, label=f"Best Fitness {column}")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_strategy_fitness_stats_from_csv(fitness_csv_filename, strategy_name = None):
    """
    Plots best, mean, std deviation, min, and max fitness values per generation for a given strategy.
    """
    df = pd.read_csv(fitness_csv_filename)
    generations = df["Generation"]

    plt.figure(figsize=(12, 8))
    plt.plot(generations, df["Mean"], label="Mean", color="blue", linestyle="--", linewidth=2)
    plt.plot(generations, df["Std"], label="Std Dev", color="orange", linestyle=":", linewidth=2)
    plt.plot(generations, df["Min"], label="Min", color="green", linewidth=2)
    plt.plot(generations, df["Max"], label="Max", color="purple", linestyle=":", linewidth=2)

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness Value", fontsize=12)
    title = f"Fitness Statistics per Generation"
    if strategy_name:
        title += f" ({strategy_name})"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_fitness_boxplot_from_csv(csv_filename, step_size=1):
    """
    Plots a boxplot of fitness values per generation from a CSV file,
    suppressing outlier points and shading the outlier area.
    """
    df = pd.read_csv(csv_filename)
    # Each column is a generation, each row is a run
    data = [df[col].dropna().values for col in df.columns]
    # Filter by step_size
    data = data[::step_size]
    
    plt.figure(figsize=(12, 8))
    box = plt.boxplot(data, patch_artist=True, showfliers=False)

    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Calculate and shade outlier areas
    for i, gen_data in enumerate(data):
        # Calculate quartiles and whiskers as matplotlib does
        q1 = np.percentile(gen_data, 25)
        q3 = np.percentile(gen_data, 75)
        iqr = q3 - q1
        lower_whisker = np.min(gen_data[gen_data >= q1 - 1.5 * iqr])
        upper_whisker = np.max(gen_data[gen_data <= q3 + 1.5 * iqr])
        min_outlier = np.min(gen_data)
        max_outlier = np.max(gen_data)
        # Shade lower outlier area if any
        if min_outlier < lower_whisker:
            plt.fill_between([i+1-0.3, i+1+0.3], min_outlier, lower_whisker, color='red', alpha=0.15)
        # Shade upper outlier area if any
        if max_outlier > upper_whisker:
            plt.fill_between([i+1-0.3, i+1+0.3], upper_whisker, max_outlier, color='purple', alpha=0.15)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title('Fitness Value Distribution Across Generations (Boxplot, Outlier Areas Shaded)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    generation_labels = [f'Gen {i}' for i in range(0, len(data)*step_size, step_size)]
    plt.xticks(range(1, len(generation_labels) + 1), generation_labels, rotation=45)

    # Legend for shaded areas
    lower_patch = mpatches.Patch(color='red', alpha=0.15, label='Lower Outlier Area')
    upper_patch = mpatches.Patch(color='purple', alpha=0.15, label='Upper Outlier Area')
    plt.legend(handles=[lower_patch, upper_patch])
    plt.tight_layout()
    plt.show()

def plot_metric_for_all_strategies(csv_files, metric, strategy_labels = None):
    """
    Plots a single metric for all strategies from their respective CSV files.

    Args:
        csv_files (list of str): List of paths to CSV files, one per strategy.
        metric (str): The metric column to plot (e.g., 'mean', 'max', 'min', 'std').
        strategy_labels (list of str, optional): Labels for each strategy. Defaults to file names.
    """
    if strategy_labels is None:
        strategy_labels = [f"Strategy {i+1}" for i in range(len(csv_files))]
    
    plt.figure(figsize=(10, 6))
    
    for csv_file, label in zip(csv_files, strategy_labels):
        df = pd.read_csv(csv_file)
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in {csv_file}. Skipping.")
            continue
        plt.plot(df[metric], label = label)
    
    plt.xlabel('Generation')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} fitness over generations for all strategies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pitch_distributions_side_by_side(midi_file_1, midi_file_2, midi_file_3, titles=None):
    """
    Plots the pitch distribution (histogram) for three MIDI files side by side.
    """
    midi_files = [midi_file_1, midi_file_2, midi_file_3]
    if titles is None:
        titles = [f"File {i+1}" for i in range(3)]
    plt.figure(figsize=(18, 5))
    for i, (midi_path, title) in enumerate(zip(midi_files, titles)):
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        pitches = [note.pitch for note in midi_data.instruments[0].notes]
        plt.subplot(1, 3, i+1)
        plt.hist(pitches, bins=range(21, 91), color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('MIDI Pitch')
        plt.ylabel('Count')
        plt.xlim(21, 90)
    plt.tight_layout()
    plt.show()

def plot_duration_distributions_side_by_side(midi_file_1, midi_file_2, midi_file_3, titles=None):
    """
    Plots the note duration distribution (histogram) for three MIDI files side by side.
    Uses a fixed number of bins and formats x-axis to show 4 decimals for better resolution.
    """
    midi_files = [midi_file_1, midi_file_2, midi_file_3]
    if titles is None:
        titles = [f"File {i+1}" for i in range(3)]
    plt.figure(figsize=(18, 5))
    for i, (midi_path, title) in enumerate(zip(midi_files, titles)):
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        durations = [note.end - note.start for note in midi_data.instruments[0].notes]
        durations = [d for d in durations if d > 0]
        plt.subplot(1, 3, i+1)
        if durations:
            plt.hist(durations, bins=30, color='lightgreen', edgecolor='black')
            plt.xlabel('Note Duration (seconds)')
            plt.ylabel('Count')
            plt.title(title)
            # Format x-ticks to 4 decimals for better resolution
            locs, labels = plt.xticks()
            if i == 0:
                plt.xticks(locs, [f"{x:.16f}" for x in locs], rotation=45)
            else:
                plt.xticks(locs, [f"{x:.2f}" for x in locs])
        else:
            plt.text(0.5, 0.5, "No notes found", ha='center', va='center', fontsize=14)
            plt.xlim(0, 2)
            plt.xticks([0, 0.5, 1, 1.5, 2], [f"{x:.2f}" for x in [0, 0.5, 1, 1.5, 2]])
            plt.title(title)
            plt.xlabel('Note Duration (seconds)')
            plt.ylabel('Count')
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

# a, b = evolve_single(1000, 500, 75, 4, 42, mode = "duration")

# print(min(a[0]))
# print(max(a[0]))
# print(b[1])


# csv_files = [
#     'fitness_stats_strategy1_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy2_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy3_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy4_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy5_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy6_30runs_500pop_75ind_1000gen.csv',
#     'fitness_stats_strategy7_30runs_500pop_75ind_1000gen.csv'
# ]
# plot_metric_for_all_strategies(csv_files, metric = 'Std')

mode = 'duration'

# multi_strategy_test(1, 1000, 500, 75, base_seed = 42, mode = mode)    

# plot_multi_strategy_test_from_csv("duration_fitness_best_all_1runs_500pop_75ind_1000gen.csv")
# plot_strategy_fitness_stats_from_csv("fitness_stats_strategy5_30runs_500pop_75ind_1000gen.csv")

# evolve_multi(1, 0, 100, 500, 50, 0, mode = mode, base_seed = 42,
#              midi_filename = f"{mode}_best_evolution_output_strategy1_1runs_500pop_50ind_100gen",
#              timing_csv_filename = f"{mode}_timing_stats_indsize50_popsize500_gens100_strategy1_runs1.csv",
#              fitness_csv_filename = f"{mode}_fitness_stats_indsize50_popsize500_gens100_strategy1_runs1.csv")
             #boxplot_csv_filename = "aggregated_fitness_indsize50_popsize500_gens1000_strategy1_runs30.csv")


# midi_file_duration = "../midi_files/duration_best_evolution_output_strategy1_1runs_500pop_50ind_100gen.mid"
# midi_file_rhythm = "../midi_files/rhythm_best_evolution_output_strategy1_1runs_500pop_50ind_100gen.mid"

# mel1_durations = midi_to_duration_sequence(midi_input_1)
# mel2_durations = midi_to_duration_sequence(midi_input_2)

# print(len(mel1_durations))
# print(min(mel1_durations), max(mel1_durations))
# print(len(mel2_durations))
# print(min(mel2_durations), max(mel2_durations))

# duration_seq = midi_to_duration_sequence(midi_file_duration)

# print(len(duration_seq))
# print(min(duration_seq), max(duration_seq))

# rhythm_seq = midi_to_rhythm_sequence(midi_file_rhythm)

# combine_evolved_sequences_to_midi(None, mel1_durations, None, midi_input_1, "duration_rhythm_combined")

# for i in range(7):
#     midi_file_pitch = f"../midi_files/pitch_best_evolution_output_strategy{i+1}_30runs_500pop_75ind_1000gen.mid"
#     midi_file_duration = f"../midi_files/duration_best_evolution_output_strategy{i+1}_1runs_500pop_75ind_1000gen.mid"
#     output_name = f"combined_best_evolution_output_strategy{i+1}_30runs_500pop_75ind_1000gen.mid"

#     pitch_seq = midi_to_relative_pitch_sequence(midi_file_pitch)
#     duration_seq = midi_to_duration_sequence(midi_file_duration)

#     combine_evolved_sequences_to_midi(pitch_seq, duration_seq, None, midi_input_1, output_name)

midi_best_pitch = "../midi_files/duration_best_evolution_output_strategy5_30runs_500pop_75ind_1000gen.mid"
midi_best_duration = "../midi_files/duration_best_evolution_output_strategy5_1runs_500pop_75ind_1000gen.mid"

plot_duration_distributions_side_by_side(midi_input_1, midi_input_2, midi_best_duration, titles = ["Overworld (Mario Song)", "Let It Be", "Strategy 5 Evolved Result"])