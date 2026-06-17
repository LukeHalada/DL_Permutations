import numpy as np
import matplotlib.pyplot as plt
class TSP():
    def __init__(self, cities_count, min=0, max=10, start=0, seed=0):
        """Generates random coordinates for cities.

        Args:
            cities_count (int, optional): Number of cities to generate. Defaults to 10.
            min (int, optional): Minimum value for coordinates. Defaults to 0.
            max (int, optional): Maximum value for coordinates. Defaults to 10.
            start (int, optional): Starting city index. All solutions will keep cities up to 'start' in constant order. Defaults to 0.
            seed (int, optional): Seed to initialize random number generator. Defaults to 0.
        """

        assert start < cities_count, "Starting city index must be less than the number of cities"
        assert min < max, "Minimum value must be less than the maximum value"
        
        self.cities_count = cities_count
        self.min = min
        self.max = max
        self.seed = seed
        self.start = start

        self._generate_random_cities()

        self.start_distance = 0
        if self.start > 0:
            self.start_distance = self.total_distance(np.arange(self.start))

            

    def _generate_random_cities(self):
        if self.seed:
            np.random.seed(self.seed)

        self.cities = np.random.uniform(self.min, self.max, (self.cities_count, 2))

    
    def get_cities_from_perm(self, perm):
        return self.cities[perm]        
    

    def total_distance(self, solution):
        """Calculates the total sum over the distances between each successive pair of cities in the sequence.
           Solution should be represented as a sequence of points.

        Args:
            solution (numpy.ndarray): A 2D array representing the solution.
            
        Returns:
            float: The total distance of the solution.
        """
        cities = np.vstack([self.cities[:self.start],
                          self.get_cities_from_perm(solution)])

        successive_diffs = cities[1:] - cities[:-1]
        successive_distance = np.sum(np.linalg.norm(successive_diffs, axis=1))

        closing_diff = cities[0] - cities[-1]
        closing_distance = np.linalg.norm(closing_diff)

        total_cicle_distance = successive_distance + closing_distance

        return total_cicle_distance
    


    def generate_solutions(self, number_of_solutions=50):
        """Generates random solutions.

        Args:
            number_of_solutions (int, optional): Number of solutions to be generated. Defaults to 50.

        Returns:
            np.ndarray: An array of solutions
        """

        perm_len = self.cities_count - self.start
        solutions = np.array([
            np.random.permutation(np.arange(perm_len))
            for _ in range(number_of_solutions)
        ])
        return solutions + self.start


    def plot_solution(self, solution, figsize=(10, 10), show=True, filename=None):
        """Plots a solution to the TSP problem.

        Args:
            solution (numpy.ndarray): A 2D array representing the solution.
            figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
            filename (str, optional): The filename to save the plot. If not specified, the plot will not be saved. Defaults to None.
        """
        solution = self.get_cities_from_perm(solution)
        plt.figure(figsize=figsize)

        # Plot starting cities
        plt.scatter(self.cities[:self.start, 0], self.cities[:self.start, 1], color='green', label='Starting city')
        for i in range(self.start):
            plt.annotate(str(i), (self.cities[i, 0], self.cities[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        for i in range(self.start - 1):
            plt.plot([self.cities[i, 0], self.cities[i+1, 0]],
                     [self.cities[i, 1], self.cities[i+1, 1]], color='green', linestyle='--')
        
        # Plot other cities
        plt.scatter(self.cities[self.start:, 0], self.cities[self.start:, 1], color='blue', label='Cities')
        for i in range(self.start, len(self.cities)):
            plt.annotate(str(i), (self.cities[i, 0], self.cities[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.plot(solution[:, 0], solution[:, 1], color='red', label='Solution')

        #Connect starting city to first and last city in solution
        if self.start > 0:
            plt.plot([self.cities[self.start-1, 0], solution[0, 0]], 
                     [self.cities[self.start-1, 1], solution[0, 1]], 
                    color='green', linestyle='--')
            plt.plot([self.cities[0, 0], solution[-1, 0]], 
                     [self.cities[0, 1], solution[-1, 1]], 
                    color='green', linestyle='--')

        else:
            plt.plot([solution[0, 0], solution[-1, 0]], 
                     [solution[0, 1], solution[-1, 1]], 
                    color='red')

        plt.title("TSP Solution")
        plt.legend()
        
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()

