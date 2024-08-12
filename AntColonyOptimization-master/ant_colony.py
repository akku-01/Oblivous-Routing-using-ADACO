import random as rn
import numpy as np
from numpy.random import choice as np_choice

class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
      
        distances : Square matrix of distances. Diagonal is assumed to be np.inf.
        n_ants: Number of ants running per iteration
        n_best: Number of best ants who deposit pheromone
        n_iteration: Number of iterations
        decay: Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
        alpha: exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
        beta: exponent on distance, higher beta give distance more weight. Default=1

        Example:
            ant_colony = AntColony(distances, 100, 20, 2000, 0.95, alpha=1, beta=2)          
        """
        self.distances  = distances
        # self.pheromone = np.ones(2) / len(distances)
        self.pheromone = np.ones(self.distances.shape) / len(distances) # is case me 0.2 se initialize hora hai bcz 1/5 
        self.gt = np.zeros(self.distances.shape) 
        self.theta = np.zeros(self.distances.shape)
        self.acc = np.zeros(self.distances.shape)
        self.update = np.zeros(self.distances.shape)
        self.d_acc = np.zeros(self.distances.shape)
        self.beta1 = (np.ones(self.distances.shape))*0.5
        self.pheromone_max = (np.ones(self.distances.shape))* 0.180
        self.pheromone_min = (np.ones(self.distances.shape))*0.050
        self.epsilon = (np.ones(self.distances.shape))*0.0001
        
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        # ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        
        # distances = np.array([[np.inf, 2, 2, 5, 7],
        #               [2, np.inf, 4, 8, 2],
        #               [2, 4, np.inf, 1, 3],
        #               [5, 8, 1, np.inf, 2],
        #               [7, 2, 3, 2, np.inf]])

        # ant_colony = AntColony(distances, 1, 1, 100, 0.95, alpha=1, beta=1)
        # shortest_path = ant_colony.run()
        # print ("shorted_path: {}".format(shortest_path))
        # print(ant_colony)

    def run(self):
        shortest_path = None   #initially there is no shortest path available
        all_time_shortest_path = ("placeholder", np.inf) 
        
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths() 
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone = self.pheromone * self.decay            
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                #Here Adaptive Gradient will be used
                # self.theta[move] = 1.0 / self.distances[move]
                # self.gt[move] = self.pheromone[move] - self.theta[move]/self.decay
                # self.acc[move] = (self.beta1[move]* self.acc[move]) + (1-self.beta1[move])*(self.gt[move])*(self.gt[move])
                # self.update[move] = (self.gt[move])*np.sqrt(self.d_acc[move] + self.epsilon[move]) / np.sqrt(self.acc[move] + self.epsilon[move])
                # self.pheromone[move] -= self.update[move]
                # self.d_acc[move] = self.beta1[move]*self.d_acc[move] + (1-self.beta1[move])*self.update[move]*self.update[move]
                # if self.pheromone[move]>self.pheromone_max[move] :
                #     self.pheromone[move] = self.pheromone_max[move]
                # if self.pheromone[move]<self.pheromone_min[move] :
                #     self.pheromone[move] = self.pheromone_min[move]
                
                #This is gradient descent normal version
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


