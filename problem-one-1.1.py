from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
from functools import partial
from collections import namedtuple
import matplotlib.pyplot as plt

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

def run_evolution(populate_func: PopulateFunc,
                  fitness_func: FitnessFunc,
                  fitness_limit: int,
                  selection_func: SelectionFunc = selection_pair,
                  crossover_func: CrossoverFunc = single_point_crossover,
                  mutation_func: MutationFunc = mutation,
                  generation_limit: int = 100) -> Tuple[Population, int]:
    population = populate_func()
    average_fitness_history = []  # To track average fitness over generations

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        average_fitness = sum([fitness_func(genome) for genome in population]) / len(population)
        average_fitness_history.append(average_fitness)

        best_fitness = fitness_func(population[0])
        print(f"Generation {i+1}: Average Fitness = {average_fitness:.2f} -- Best Fitness = {best_fitness}")
              
        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation.extend([offspring_a, offspring_b])

        population = next_generation

    # Plotting average fitness over generations
    plt.plot(average_fitness_history)
    plt.title('Average Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.show()

    return population, i


#this is an extremely easy fitness function
def fitness(genome: Genome) -> int:
    return sum(genome)


genome_length = 30
population_size = 100
fitness_limit = 30
generation_limit = 100

population, generations = run_evolution(
    populate_func=partial(generate_population, size=population_size, genome_length=genome_length),
    fitness_func=fitness,  # Note that this is now just counting 1s
    fitness_limit=fitness_limit,
    generation_limit=generation_limit
)


# Correct final print statements to match the task
best_genome = population[0]
best_fitness = fitness(best_genome)
print(f"Number of generations: {generations}")
print(f"Best Genome: {best_genome} with Fitness: {best_fitness}")

