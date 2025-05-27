#Evan Mulcare (R00211686)

import math
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Task One
def task_one():
  #Load Dataset
  data = pd.read_csv("energy_performance.csv")

  #Seperate into features and columns
  feature_columns = ['Relative compactness', 'Surface area', 'Wall area', 'Roof area', 
                    'Overall height', 'Orientation', 'Glazing area', 'Glazing area distribution']
  target_columns = ['Heating load', 'Cooling load']

  #convert to numpy arrays for easier usage
  features = data[feature_columns].to_numpy()
  targets = data[target_columns].to_numpy()

  #identify and print the minimum / maximum for heating and cooling loads.
  heating_load = data['Heating load']  
  cooling_load = data['Cooling load']

  print(f"Minimum Heating Load: {heating_load.min()}")
  print(f"Minimum Cooling Load: {cooling_load.min()}")

  print(f"Maximum Heating Load: {heating_load.max()}")
  print(f"Maximum Cooling Load: {cooling_load.max()}")

  #return features and targets for later usage
  return features, targets

# Task Two
def task_2_calculate_polynomial_model(features, deg, coefficients):
  n_samples = features.shape[0] # number of rows in the input data
  estimated_target = np.zeros(n_samples) #Initalize estimated target vector results array
  coefficient_index = 0 #keep track of coefficient index 
  
  #loop through all polynomial degrees
  for degree in range(deg + 1):
    #itertools.product generates all combinations of powers for the features
    #generating combinations for the 8 features
    for powers in itertools.product(range(degree + 1), repeat=8):
        # only use combinations where the sum degree of the polynomial is equal to the current degree
        if sum(powers) == degree:
          term = 1
          for feature_index in range(8): 
              term *= features[:, feature_index] ** powers[feature_index] #apply power to each feature
          #add result weighted by the coefficient
          estimated_target += coefficients[coefficient_index] * term
          #increment coeffiecient
          coefficient_index += 1  

  #returns the estimated target vector
  return estimated_target

def task_two_calculate_parameter_vector_size(number_of_features, deg):
  # Calculate the parameter vector size needed for the model based on the degree and number of features
  # using the binomial coefficient formula C(n+d,d) - number of combinations of powers for the features
  return math.comb(number_of_features + deg, deg)

#Task 3
def task_three_linearization(features, deg, coefficients):
   #number of samples from input data
   n_samples = features.shape[0]
   #Initialize the Jacobian matrix
   jacobian = np.zeros((n_samples, len(coefficients)))
   
   #IMPORTANT - model function froim task 2 called here.
   #WHY - call model function implemented in task 2 to get the estimated target values at the linearization point
   #called here as the model vale at the current coefficient values is necessary to compare against the pertubed model values later
   model_value = task_2_calculate_polynomial_model(features, deg, coefficients)
   epsilon = 1e-6 #used to get derivative rate of change between coeffiecients

   #loop over all coefficients
   for i in range(len(coefficients)):
      temporary_coefficients = coefficients.copy()
      #add the epsilon to the i-th coefficient to find the derivative
      temporary_coefficients[i] += epsilon
      
      #IMPORTANT - model function froim task 2 called here.
      #WHY - calculate the model function again, this time with the new coefficients
      #called to get the model value with the modified coefficient values
      temporary_model_value = task_2_calculate_polynomial_model(features,deg, temporary_coefficients)
    
      #IMPORTANT - partial derivaties for jacobian calulcated here
      #HOW - calulate the partial derivative by finding the difference between the pertubed model value and the original model value.
      #get the rate of change of the model for the i-th coefficient
      partial_derivative = (temporary_model_value - model_value) / epsilon
      #store this derivative in the jacobian matrix
      jacobian[:, i] = partial_derivative

   #return the jacobian and the model value
   return jacobian, model_value

#Task 4
def task_four_update(jacobian, model_value, targets):
 regularization_param = 1e-2 #regularization used to prevent overfitting
 
 #IMPORTANT - residuals calculated here.
 #HOW - the difference between the actual target values and the estimated target values gives the residuals.
 residuals = targets - model_value 

 #IMPORTANT - normal equation matrix caluclated here.
 #HOW - normal equation matrix created as Jacobian^T * Jacobian this is added with the regularization_param to prevent overfitting. 
 normal_equation_matrix = np.matmul(jacobian.T, jacobian) + regularization_param * np.eye(jacobian.shape[1])
 right_side_normal_equation = np.matmul(jacobian.T, residuals) # create right-hand side of normal equation. 

 #solve the normal equation to get the optimal update value-that is the change that needs to be applied to the current coefficients
 update_value = np.linalg.solve(normal_equation_matrix, right_side_normal_equation)
 return update_value

#task 5
def task_five_regression(deg, features, targets):
  max_iter = 10 #number of iterations to use in the regrssion

  #find the optimal size of the parameter vector based on the polynomial degree by calling the function created in task 2
  parameter_vector_size = task_two_calculate_parameter_vector_size(8, deg)
  
  #IMPORTANT - this is the parameter vector
  #HOW IS IT UPDATED? it is updated iteratively below by calling the linearization and update tasks and append the update value to the parameter vector
  parameter_vector = np.zeros(parameter_vector_size)

  #update the parameter_vector to reduce error
  for i in range(max_iter):
    #get the jacobian matrix and estimated target values by calling the linearization function from task 3
    jacobian, model_value = task_three_linearization(features, deg, parameter_vector)

    #find the update value for the parameter_vector using the update function from task 4
    update_value = task_four_update(jacobian, model_value, targets)
    #apply the update value to the parameter_vector
    parameter_vector += update_value

  #return the parameter_vector after updating
  return parameter_vector

#Task 6
def task_six_cross_validation(features, targets, deg, k=5):
   #Initialize K-Folds cross-validation with n_splits set to the k input parameter
   kf = KFold(n_splits=k, shuffle=True, random_state=42)

   #initalize empty arrys to store results for heating, cooling and a variable to track the current fold.
   heating_difference_results = []
   cooling_difference_results = []
   fold = 1

   for train_index, test_index in kf.split(features):
      #split into training and validation sets
      X_train, X_test = features[train_index], features[test_index]
      y_train, y_test = targets[train_index], targets[test_index]

      #perform regression to get optimal coefficients
      heating_coefficients = task_five_regression(deg, X_train, y_train[:, 0])
      cooling_coefficients = task_five_regression(deg, X_train, y_train[:, 1])
      #gets estimates of heating and cooling targets
      heating_estimate = task_2_calculate_polynomial_model(X_test, deg, heating_coefficients)
      cooling_estimate = task_2_calculate_polynomial_model(X_test, deg, cooling_coefficients)
      #find the absolute differences between predicted and actual target values
      heating_abs_diff = np.abs(heating_estimate - y_test[:, 0])
      cooling_abs_diff = np.abs(cooling_estimate - y_test[:, 1])
      #get mean absolute difference
      heating_difference = np.mean(heating_abs_diff)
      cooling_difference = np.mean(cooling_abs_diff)
      #output differences for current fold.
      print(f"Fold {fold}: Heating Difference = {heating_difference:.4f}, Cooling Difference = {cooling_difference:.4f}")
      #append to output arrays
      heating_difference_results.append(heating_difference)
      cooling_difference_results.append(cooling_difference)   
      fold += 1                  

   #get the mean absolute difference for both heating and cooling kFolds cross validation procedures.
   average_difference_heating = np.mean(heating_difference_results)
   average_difference_cooling = np.mean(cooling_difference_results)
  
   return average_difference_heating, average_difference_cooling
 
def task_six_find_optimal_polynomial_degree(features, targets):
    cross_validation_results = []
    for degree in range(3):
        print(f"\nRUNNING CROSS VALIDATION FOR DEGREE {degree}...")

        # Perform cross-validation and get error values
        avg_diff_heating, avg_diff_cooling = task_six_cross_validation(features, targets, degree)
        print(f"Degree {degree} - Heating Mean Absolute Error: {avg_diff_heating:.4f}, Cooling Mean Absolute Error: {avg_diff_cooling:.4f}")
        cross_validation_results.append((degree, avg_diff_heating, avg_diff_cooling))

    # choose the optimal degree as the value the smallest combined error
    optimal_degree = min(cross_validation_results, key=lambda errors: errors[1] + errors[2])[0]
    print(f"\nOptimal Polynomial Degree: {optimal_degree}")

    return optimal_degree

def task_seven(features, targets, optimal_degree):
  #perform regression to get optimal coefficients
  heating_coefficients = task_five_regression(optimal_degree, features, targets[:, 0])
  cooling_coefficients = task_five_regression(optimal_degree, features, targets[:, 1])
  #gets estimates of heating and cooling targets
  predicted_heating = task_2_calculate_polynomial_model(features, optimal_degree, heating_coefficients)
  predicted_cooling = task_2_calculate_polynomial_model(features, optimal_degree, cooling_coefficients)
  #find the absolute differences between predicted and actual target values
  heating_abs_diff = np.abs(predicted_heating - targets[:, 0])
  cooling_abs_diff = np.abs(predicted_cooling - targets[:, 1])
  #get mean absolute difference
  heating_difference = np.mean(heating_abs_diff)
  cooling_difference = np.mean(cooling_abs_diff)
  
  print(f"Mean Absolute Difference (Heating): {heating_difference:.4f}")
  print(f"Mean Absolute Difference (Cooling): {cooling_difference:.4f}")

  # plot heating load graph
  plt.subplot(1,2,1)
  task_seven_plots(predicted_heating,targets[:, 0],"Heating")

  # plot cooling load graph
  plt.subplot(1,2,2)
  task_seven_plots(predicted_cooling,targets[:, 1],"Cooling")

  plt.suptitle("Estimated vs True Loads", fontsize=16)  
  plt.tight_layout()
  plt.show()

def task_seven_plots(estimated_loads, true_loads, kind):
    #scatter plot of true loads vs estimated loads
    plt.scatter(true_loads, estimated_loads, alpha=0.5, color='green',
        edgecolor='black', label=f"{kind} Data Points")
    #get minimum and maximum values of the true loads for the perfect fit line
    min_value = true_loads.min()
    max_value = true_loads.max()
    # perfect fit line
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label="Perfect Fit Line")

    plt.xlabel(f"True {kind} Load")
    plt.ylabel(f"Estimated {kind} Load")

    plt.title(f"{kind}")
    plt.legend()
    plt.grid(color='gray', linestyle=':', linewidth=0.7)

def main():
    features, targets = task_one()
    optimal_degree = task_six_find_optimal_polynomial_degree(features, targets)
    print(f"\nRUNNING TASK 7,MODEL EVALUATION WITH OPTIMAL DEGREE {optimal_degree}...")
    task_seven(features, targets, optimal_degree)

if __name__ == "__main__":
    main()