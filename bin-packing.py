import matplotlib.pyplot as plt
import numpy as np
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
from functools import partial
from collections import namedtuple


# Generates the initial population randomly for the bin packing problem.
def generate_population(size):
    population = []
    for _ in range(size):
        Genome = [[] for _ in range(DATA_INSTANCE)]
        for item in DATA:
            random_index = np.random.choice(DATA_INSTANCE)
            Genome[random_index].append(item)
        population.append(Genome)
    return np.array(population)

# Returns the mean fitness of the population
def population_fitness(population):
    return np.mean([fitness(item) for item in population])

# Performs the crossover operation between two genomes
# crossover point picked randomly 
# returns the two new child genomes
def single_point_crossover(GenomeA, GenomeB):
    length = len(GenomeA)
    p = randint(1, length - 1)
    GenomeA = np.concatenate((GenomeA[:p], GenomeB[p:]))
    GenomeB = np.concatenate((GenomeB[:p], GenomeA[p:]))
    return GenomeA, GenomeB

# Performs the mutation operation on the genome
# randomly selects a bin and moves an item to another bin
# returns the mutated genome
def mutation(Genome):
    for bin in Genome:
        index = randrange(len(Genome))
        if index and len(bin) > 0:
            genome_location = bin.pop(randrange(len(bin)))
            new_gemone_location = randint(1, len(Genome) - 1)
            Genome[new_gemone_location].append(genome_location)
    return Genome

# Runs the genetic algorithm for the bin packing problem
def run_evolution(generation_limit, size):
    # create the starting population 
    population = generate_population(size) 
    # first generation fitness, random data fitness
    generation_fitness = [population_fitness(population)]
    print(f"Fitness at generation 0: {population_fitness(population)}")

    # loop through the generations
    for i in range(generation_limit):
        # calculate the fitness of the population
        fitness_score = np.array([fitness(Genome) for Genome in population])
        # sort the population by fitness and select the top half
        sorted = np.argsort(fitness_score)[::-1]
        best_genome = population[sorted[:size//2]]

        # store next generation
        Next_Generation = []
        # performing crossover and mutation to make next generation
        for j in range(len(best_genome)):
            GenomeA, GenomeB = best_genome[np.random.choice(len(best_genome), size=2, replace=False)]
            OffspringA, OffspringB = single_point_crossover(GenomeA, GenomeB)
            OffspringA = mutation(OffspringA)
            OffspringB = mutation(OffspringB)
            Next_Generation.extend([OffspringA, OffspringB])

        # replace the old population with the new generation
        population = np.array(Next_Generation)
        # for printing and plotting 
        generation_fitness.append(population_fitness(population))
        print(f"Fitness at generation {i+1}: {population_fitness(population)}, Bins: {bins_required(population[np.argmax(fitness_score)])}")

    # printing the fittest genome and the number of bins used
    np.set_printoptions(threshold=np.inf)
    fittest_genome = population[np.argmax(fitness_score)]
    print(best_genome)
    bins_used = bins_required(fittest_genome)
    print(f"Bins: {bins_used}")
    # Create a new figure
    plt.figure(figsize=(10, 6))
    plt.plot(range(generation_limit+1), generation_fitness, marker="o", color="red")
    # Set the title and labels with increased font sizes
    plt.title("Generation Fitness", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Mean Population Fitness", fontsize=14)
    # Add a grid for better readability
    plt.grid(True)  
    #Add a legend to clarify what the line represents
    plt.legend(['Fitness'], loc='upper left')
    plt.show()

# Returns the number of bins required to store the items
def bins_required(Genome):
    countBins = 0
    for bin in Genome:
        if np.sum(bin) > 0:
            countBins += 1
    return countBins

# Returns the fitness of the genome
def fitness(Genome):
    # number of items not in bin
    data_omitted = 0
    # copy of data
    data_copy = DATA.copy()
    for bin in Genome:
        # if item not in data, increment data_omitted
        for item in bin:
            if item not in data_copy:
                data_omitted +=1
            else:
                # remove item from data if present
                data_copy.remove(item)
    # add the number of items not in bin to data_omitted
    data_omitted += len(data_copy)

    # adds penalties when bins are not properly filled
    # as well as considering data omitted from the bins.
    bins_underflowed = 0
    for bin in Genome:
        # Check if the bin is not empty and its total weight is below the maximum allowed weight
        if np.sum(bin) > 0 and np.sum(bin) < MAX_BIN_WEIGHT:
            # represents the wasted capacity
            bins_underflowed += MAX_BIN_WEIGHT - np.sum(bin)

    bins_overflowed = 0
    for bin in Genome:
        # Check if the bin's total weight exceeds the maximum allowed weight
        if np.sum(bin) > 0 and np.sum(bin) > MAX_BIN_WEIGHT:
            # represents the overflowed capacity
            bins_overflowed += np.sum(bin) - MAX_BIN_WEIGHT

    # returns a negative value, penalizing solutions based on their overflow and underflow
    # penalises 'data_omitted' ie. count of items not included in any bin
    # multiplied account for the severity of larger items
    return -bins_overflowed-bins_underflowed-(data_omitted*max(DATA))

# Loads the data from the file
def load_data(filename):
    content = open(filename).read().split("\n")
    max_bin_capacity = int(content[2])
    items = []
    for line in content[3:]:
        number, times = [int(x) for x in line.split(" ")]
        items.extend([number for _ in range(times)])
    num_items = len(items)
    return max_bin_capacity, items, num_items

# Main function - run different data instances
if __name__ == '__main__':
    generation_limit = 100
    size = 1000
    for i in range(1, 6):
        filename = f"binpacking_data/data_{i}.txt"
        MAX_BIN_WEIGHT, DATA, DATA_INSTANCE = load_data(filename)
        run_evolution(generation_limit, size)


