import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

# TASK ONE
def task_one():
    data = pd.read_csv("fashion-mnist_train.csv")
    filtered_data = data.query('label in [5, 7, 9]')
    
    labels = filtered_data['label']
    features = filtered_data.drop('label', axis=1)
    
    features = pd.DataFrame(StandardScaler().fit_transform(features), columns=features.columns) 
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    
    class_names = {5: 'Sandal', 7: 'Sneaker', 9: 'Ankle boot'}
    display_images = {}
    for label in class_names:
        label_indices = labels[labels == label].index
        first_index = label_indices[0]
        display_images[label] = features.iloc[first_index]

    figure, image_axes = plt.subplots(1, 3, figsize=(10, 4)) 
    for axis, (product_label, image_pixels) in zip(image_axes, display_images.items()):
        image = image_pixels.values.reshape(28, 28) 
        axis.imshow(image, cmap='gray')  
        axis.set_title(f"{class_names[product_label]} - {product_label}")  

    figure.suptitle("Image Display for Each class", fontsize=16)
    plt.show()

    return features, labels
features, labels = task_one()


def task_two(features, labels, classifier, n_splits, sample_size=len(features), verbose=False):
    sampled_indices = np.random.choice(len(features), sample_size, replace=False)
    
    sampled_features = features.iloc[sampled_indices]  
    sampled_labels = labels.iloc[sampled_indices] 

    images_array = sampled_features
    label_array = sampled_labels.to_numpy() 

    training_times = []
    prediction_times = []
    accuracies = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    
    if verbose:
        print(f"\nSTARTING K-FOLD CROSS VALIDATION FOR {classifier.__class__.__name__}: \n")

    for training_index, validation_index in kf.split(images_array):
        # Split into training and validation sets
        training_images, validation_images = images_array.iloc[training_index], images_array.iloc[validation_index]
        training_labels, validation_labels = label_array[training_index], label_array[validation_index]

        if verbose:
            print(f"CURRENT FOLD : {fold}: ")
            print(f"Training set size: {len(training_images)}, Validation set size: {len(validation_images)}")
        
        start_time_training = time.time()
        classifier.fit(training_images, training_labels)
        end_time_training = time.time()
        training_time = end_time_training - start_time_training

        start_time_prediction = time.time()
        validation_predictions = classifier.predict(validation_images)
        end_time_prediction = time.time()
        prediction_time = end_time_prediction - start_time_prediction

        accuracy = accuracy_score(validation_labels, validation_predictions)
        c_matrix = confusion_matrix(validation_labels, validation_predictions)

        training_times.append(training_time)
        prediction_times.append(prediction_time)
        accuracies.append(accuracy)
        
        results = []
        results.append([
            f"Fold {fold}",
            f"{training_time:.4f} seconds",
            f"{prediction_time:.4f} seconds",
            f"{accuracy:.4f}",
            f"{c_matrix}"
        ])

        headers = ["Fold", "Training Time", "Prediction Time", "Accuracy", "Confusion Matrix"]
        
        if verbose:
            print(tabulate(results, headers, tablefmt="grid"))
        fold += 1

    training_times_per_sample = []
    prediction_times_per_sample = []
    for entry in training_times:
        training_times_per_sample.append(entry / len(training_images))

    for entry in prediction_times:
        prediction_times_per_sample.append(entry / len(validation_images))

    print(f"\nEND RESULT FOR  {classifier.__class__.__name__} CLASSIFIER : SUMMARY OF K-FOLD CROSS VALIDATION\n")
    summary_results = [
        ["Training time per sample"] + list(calculate_metrics(training_times_per_sample)),
        ["Prediction time per sample"] + list(calculate_metrics(prediction_times_per_sample)),
        ["Accuracy"] + list(calculate_metrics(accuracies))
    ]
    summary_headers = ["Metric", "Min", "Max", "Avg"]
    print(tabulate(summary_results, summary_headers, tablefmt="grid", floatfmt=".6f"))

    return accuracies, training_times, prediction_times

#HELPER FOR QUESTION 2 RESULTS
def calculate_metrics(values):
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    return min_val, max_val, avg_val

#HELPER FOR EVALUATION OF SAMPLE SIZES + PLOTTING
def evaluate_by_sample_size(features,labels,classifier):
   sample_sizes = [100, 500, 1000, 2000, 5000, 10000, len(features)]
   sample_size_training_times = []
   sample_size_prediction_times = []
   sample_size_accuracies = []

   for sample_size in sample_sizes:
      print(f"\nEvaluating with {sample_size} samples...")
      sample_accuracy, sample_training_time, sample_prediction_time = task_two(features, labels, classifier, n_splits=5, sample_size=sample_size)

      mean_accuracy = np.mean(sample_accuracy) 
      mean_training_time = np.mean(sample_training_time)  
      mean_prediction_time = np.mean(sample_prediction_time) 

      sample_size_accuracies.append(mean_accuracy)
      sample_size_training_times.append(mean_training_time)
      sample_size_prediction_times.append(mean_prediction_time)
   
   #Mean accuracies accross all sample sizes.
   print(f"\nMean Prediction Accuracy for {classifier.__class__.__name__}: {np.mean(sample_size_accuracies):.4f}")

   #PLOT 
   fig, ax = plt.subplots(figsize=(10, 6))
    # Training Time
   ax.plot(sample_sizes, sample_size_training_times, marker='x', linestyle='-', color='green', label='Training Time')
    # Prediction Time
   ax.plot(sample_sizes, sample_size_prediction_times, marker='d', linestyle=':', color='red', label='Prediction Time')

   ax.set_title(f"Training and Prediction Time vs Sample Size ({classifier.__class__.__name__})", fontsize=14, fontweight='bold')
   ax.set_xlabel("Sample Size", fontsize=12)
   ax.set_ylabel("Time (seconds)", fontsize=12)

   ax.legend(fontsize=10, loc="upper left")
   plt.show()

#Task 3
perceptron_classifier = Perceptron(random_state=42)
evaluate_by_sample_size(features,labels,perceptron_classifier)

#Task 4
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
evaluate_by_sample_size(features,labels,decision_tree_classifier)

#task 5
def task_five(features, labels):
  best_accuracy = 0
  best_k = 0

  for k in range(1, 11): 
    print(f"\nEvaluating k-NN classifier with k={k}...")
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    sample_accuracy, _, _ = task_two(features, labels, knn_classifier, n_splits=5)

    mean_accuracy = np.mean(sample_accuracy)
    print(f"Mean accuracy for k={k}: {mean_accuracy:.4f}")
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

  print(f"\n The best achievable mean prediction accuracy for K was: {best_accuracy:.4f} with a  K value of: {best_k}")

  print(f"\nEvaluating k-NN classifier with optimal k value : {best_k}...")
  knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
  evaluate_by_sample_size(features, labels, knn_classifier)

task_five(features, labels)

# TASK SIX
def task_six(features, labels):
    gamma_values = [0.01, 0.1, 1, 10]
    best_gamma = None
    best_accuracy = 0

    print("Evaluating SVM with different γ(gamma) values...\n")
    for gamma in gamma_values:
        print(f"\nTesting SVM with gamma = {gamma}")
        svm_classifier = svm.SVC(kernel="rbf", gamma=gamma)
        print(f"\n Running Task two to train and evaluate SVM with gamma = {gamma}\n")
        accuracies, _, _ = task_two(features, labels, svm_classifier, n_splits=5, verbose=True)
        print("\nTASK TWO COMPLETE\n")
        mean_accuracy = np.mean(accuracies) 
        print(f"Mean accuracy for gamma={gamma}: {mean_accuracy:.4f}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_gamma = gamma

    print(f"\n The best achievable mean prediction accuracy for gamma was: {best_accuracy:.4f} with a gamma value of: {best_gamma}")
    svm_classifier = svm.SVC(kernel="rbf", gamma=best_gamma)
    evaluate_by_sample_size(features, labels, svm_classifier)

task_six(features, labels)