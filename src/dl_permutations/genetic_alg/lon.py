import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from collections import defaultdict
from .tsp import TSP

def generate_local_optima_network(
    local_optima: list, 
    ga_instance, 
    tsp_instance: TSP,
    samples_per_node: int = 50, 
    pop_size: int = 20, 
    epochs: int = 50,
) -> nx.DiGraph:
    """
    Generates a Local Optima Network (LON) directed graph.
    
    Args:
        local_optima (list): List of tuples/arrays representing known local minima.
        ga_instance: An initialized instance of your GA class.
        tsp (TSP): An initialized instance of TSP class.
        samples_per_node (int): How many GA runs to sample per optimum to find probabilities.
        pop_size (int): Size of the starting population for each sample run.
        epochs (int): Number of generations to run the GA for each sample.
        
    Returns:
        nx.DiGraph: A directed graph where nodes are optima and edges carry 'weight' (probability).
    """

    lon = nx.DiGraph()
    
    for i, optimum in enumerate(local_optima):
        lon.add_node(i, permutation=optimum)
    
    #Iterate through optima
    for i, current_optimum in enumerate(tqdm(local_optima, desc="Processing Optima")):
        transition_counts = defaultdict(int)
        
        #Generate sample ga runs
        for _ in range(samples_per_node):
            
            #Generate starting population
            starting_population = []
            for _ in range(pop_size):
                mutant = ga_instance.mutation(np.array(current_optimum).copy())
                starting_population.append(mutant)
            
            starting_population = np.array(starting_population)
            
            #Fit
            final_pop, best_history = ga_instance.fit(
                epochs=epochs, 
                starting_population=starting_population, 
                verbose=-1
            )
            
            best_solution_found = min(final_pop, key=tsp_instance.total_distance)
            
            # Find closest optimum
            closest_optimum_idx = i 
            for j, candidate_optimum in enumerate(local_optima):
                if np.array_equal(best_solution_found, candidate_optimum):
                    closest_optimum_idx = j
                    break
                    
            transition_counts[closest_optimum_idx] += 1
            
        #Calculate probabilities
        for target_idx, count in transition_counts.items():
            probability = count / samples_per_node

            #Add edges
            if probability > 0:
                lon.add_edge(i, target_idx, weight=probability)
                
    return lon