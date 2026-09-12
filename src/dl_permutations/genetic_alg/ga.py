import numpy as np
import matplotlib.pyplot as plt

class GA_TSP():

    def __init__(self,
                 cities:np.ndarray,
                 population_size:int,
                 mutation_rate:float,
                 cross_rate:float,
                 tournament_size:int,
                 fitness:callable):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.tournament_size = tournament_size
        self.fitness = fitness
    
    def tournamet_selection(self, population:np.ndarray):
        """Preforms tournament selection by choosing the bracket at random from the population,
        then selecting the best instance.

        Args:
            population (np.ndarray): Population to choose from.

        Returns:
            np.ndarray: The best instance from the tournament.
        """
        bracket = np.random.choice(np.arange(self.population_size), size=self.tournament_size, replace=False)
        return min(population[bracket], key=self.fitness)
    
    def crossover(self, parent1:np.ndarray, parent2:np.ndarray):
        """Preforms an ordered crossover, preserving the permutation structure.

        Args:
            parent1 (np.ndarray): First parent.
            parent2 (np.ndarray): Second parent.

        Returns:
            np.ndarray: A new member of the population created from the parents.
        """
        start = np.random.randint(0, len(parent1) - 1)
        end = np.random.randint(start, len(parent1)-1)

        #first take a fragment of parent1 and put it in child
        child = np.empty(len(parent1), dtype=int)
        child.fill(-1)
        child[start:end+1] = parent1[start:end+1]

        #fill the rest of child with values from parent2
        p2_pointer = 0
        for i in range(child.shape[0]):
            if child[i] < 0:
                while parent2[p2_pointer] in child:
                    p2_pointer += 1
                child[i] = parent2[p2_pointer]
        
        return child
    
    def mutation(self, child:np.ndarray):
        """Preforms a swap mutation, swapping two elements of the child permutation.

        Args:
            child (np.ndarray): Member of a population to be mutated.

        Returns:
            np.ndarray: The mutated member of the population.
        """
        for i in range(child.shape[0]):
            if np.random.uniform(0, 1) < self.mutation_rate:
                target = np.random.randint(0, child.shape[0]-1)
                child[i], child[target] = child[target], child[i]
        return child
    
    def fit(self, 
            epochs:int, 
            starting_population:np.ndarray,
            verbose=-1):
        """Calculates new generations iteratively.

        Args:
            epochs (int): Number of generations to calculate.
            starting_population (np.ndarray): An array of initial permutations.
            verbose (int, optional): Number of generations between print statements. -1 for no print statements. Defaults to -1.

        Returns:
            np.ndarray: The final population.
            np.ndarray: The history of the best route in each generation.
        """
        population = starting_population
        history = []
        for generation in range(epochs):
            new_population = []
            best_route = min(population, key=self.fitness)
            history.append(best_route)
            new_population.append(best_route)

            if verbose > 0 and(generation + 1)%verbose == 0:
                print(f"Generation {generation+1}: {self.fitness(best_route)}")

            for _ in range(1, self.population_size):
                p1 = self.tournamet_selection(population)
                p2 = self.tournamet_selection(population)
                child = self.crossover(p1, p2)
                child = self.mutation(child)
                new_population.append(child)

            population = np.array(new_population)

        return np.array(population), np.array(history)
