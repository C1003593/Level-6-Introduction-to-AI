import numpy
import pygad

function_inputs = [4,-2,3.5,5,-11,-4.7]  
desired_output = 100  

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

num_generations = 50
num_parents_mating = 10
sol_per_pop = 50
num_genes = len(function_inputs)
init_range_low = -2
init_range_high = 5
mutation_percent_genes = 1

ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating, 
                    fitness_func=fitness_func,
                    sol_per_pop=sol_per_pop, 
                    num_genes=num_genes,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    mutation_percent_genes=mutation_percent_genes)
ga_instance.run()
ga_instance.plot_result()

#Pygad