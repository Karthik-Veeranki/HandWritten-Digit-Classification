from scipy.optimize import minimize
import neuralNetwork


def calculateMinimum(initial_nn_parameters, my_args, max_iter):
    # Calling minimize function to optimize the cost function
    final_results = minimize(neuralNetwork.neuralNetwork, x0=initial_nn_parameters,  # function output which we need to optimize
                            args=my_args,                                            # arguments passed for the function, other than initial parameters.
                            options={'maxiter': max_iter},                           # setting stopping criteria as iterations <= 100
                            method='L-BFGS-B',                                       # Limited memory optimization method used mainly for large-scale computations
                            jac=True)                                                # setting jacobian method of computing gradient vector.
    
    return final_results['x']


'''
final_results is a dictionary-like object containing various keys like:
-> x            :    solution (value of the variable that minimizes the cost function)
-> success      :    true/false indicating the success of optimization
-> message      :    description of status of optimization
-> fun          :    value of the function at the solution
-> jac          :    gradient of the function at the solution
-> hess         :    Hessian matrix
-> nfev         :    number of function evaluations
-> nit          :    number of iterations performed

'''