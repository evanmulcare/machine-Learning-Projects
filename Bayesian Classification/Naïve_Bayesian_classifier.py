import math
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
# TASK 1 - seperate data into meaningful categories for testing and evaulation
def task_one():
    df = pd.read_excel("movie_reviews.xlsx")

    #access the specified columns,store values where the split column equals parameter, drop any NA values. convert the result to a list.
    training_data = df['Review'].where(df['Split'] == 'train').dropna().tolist()
    test_data = df['Review'].where(df['Split'] == 'test').dropna().tolist()

    training_labels = df['Sentiment'].where(df['Split'] == 'train').dropna().tolist()
    test_labels = df['Sentiment'].where(df['Split'] == 'test').dropna().tolist()

    #Count and print positive and negative amount for training and test labels.
    positive_reviews_training_count = training_labels.count('positive')
    negative_reviews_training_count = training_labels.count('negative')
    positive_reviews_testing_count = test_labels.count('positive')
    negative_reviews_testing_count = test_labels.count('negative')

    print(f"There are {positive_reviews_training_count} Positive reviews in the training set.")
    print(f"There are {negative_reviews_training_count} Negative reviews in the training set.")
    print(f"There are {positive_reviews_testing_count} Positive reviews in the evaluation set.")
    print(f"There are {negative_reviews_testing_count} Negative reviews in the evaluation set.")

    return training_data, test_data, training_labels, test_labels, positive_reviews_training_count, negative_reviews_training_count

# TASK 2 - clean data and produce word-list based on defined parameters of word size and word occurrance amount. 
def task_two(training_data, word_length, word_occurrences):
    cleaned_training_data = task_two_part_a(training_data)
    filtered_words = task_two_part_b(cleaned_training_data, word_length, word_occurrences)
    return cleaned_training_data, filtered_words

def task_two_part_a(training_data): 
    #loop through all reviews in the training data and clean as specified.
    output = []
    for review in training_data:
        output.append(clean_single_review(review))
    return output

#HELPER FUNCTION - for task 2 and also used in task 5
def clean_single_review(review):
  #format review - only alphanumnerical characters and spaces, and split review into individual words.
  parsed_review = []
  for char in review:
      if char.isalnum() or char.isspace():
          parsed_review.append(char.lower())
  parsed_review = ''.join(parsed_review).split()
  return parsed_review

def task_two_part_b(training_data, word_length, word_occurrences):
    #count the frequency of words that appear in training data.
    word_counter = {}
    for review in training_data:
        for word in review:
            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1

    #append the words to the output if they satisfy the frequency and length requirements.
    output = []
    for word, count in word_counter.items():
        if len(word) >= word_length and count >= word_occurrences:
            output.append(word)
    
    return output

# TASK 3 - Map frequenices of times a word appears in the different reviews, used for both positve and negative.
def task_three(cleaned_training_data, training_labels, valid_words):
    positive_reviews = []
    negative_reviews = []

    for i in range(len(cleaned_training_data)):
        review = cleaned_training_data[i]
        label = training_labels[i]
        
        if label == 'positive':
            positive_reviews.append(review)
        elif label == 'negative':
            negative_reviews.append(review)

    # Calculate word frequencies for positive and negative reviews using task_three
    positive_word_map = task_three_part_b(positive_reviews, valid_words)
    negative_word_map = task_three_part_b(negative_reviews, valid_words)
    return positive_word_map, negative_word_map

def task_three_part_b(valid_reviews, valid_words):
    #fill output initally with all valid words and 0 count
    output = {}
    for word in valid_words:
        output[word] = 0
    
    #split reviews into set of words, get interesction of the review words and all words and increment the values in the intersection.
    for review in valid_reviews:
        review_words = set(review)
        common_words = review_words & valid_words
        for word in common_words:
            output[word] += 1
    
    return output

# TASK 4 - get likelihoods and prior values for positive / negative words and reviews.
def task_four(positive_reviews_training_count, negative_reviews_training_count, positive_word_map, negative_word_map, valid_words):
    #empty maps that will store the positive and negative likelihoods for each word.
    positive_likelihoods = {}
    negative_likelihoods = {}

    #smoothing factor = 1
    alpha = 1
    # Calculate and store the likelihoods for each word.
    for word in valid_words:
        positive_count = positive_word_map.get(word, 0)
        negative_count = negative_word_map.get(word, 0)

        # Calculate:
        # P[word is present in review | review is positive] - the probability that a word is present in a review, given that the review is positive.
        # P[word is present in review | review is negative] - the probability that a word is present in a review, given that the review is negative.

        #laplace smoothing applied through formula: P = (count + alpha) / (total_count + alpha * N)
        positive_likelihood = (positive_count + alpha) / (positive_reviews_training_count + alpha * 2)
        negative_likelihood = (negative_count + alpha) / (negative_reviews_training_count + alpha * 2)

        positive_likelihoods[word] = positive_likelihood
        negative_likelihoods[word] = negative_likelihood

    # Calculate the prior probabilities for positive and negative reviews based on their proportions in the training set.
    total_review_count = positive_reviews_training_count + negative_reviews_training_count
    positive_prior = positive_reviews_training_count / total_review_count
    negative_prior = negative_reviews_training_count / total_review_count

    return positive_likelihoods, negative_likelihoods, positive_prior, negative_prior

# TASK 5 - classify a new review using Naive Bayes
def task_five(new_review_text, positive_likelihoods, negative_likelihoods, positive_prior, negative_prior, valid_words):
    #clean the new review and seperate into set of words
    cleaned_review = clean_single_review(new_review_text)
    review_words = set(cleaned_review)

    #Initialize the probabilities based on prior values - logs used to prevent underflow
    positive_log_probability = math.log(positive_prior)
    negative_log_probability = math.log(negative_prior)

    #get intersection of words in both the review and all valid words.
    common_words = review_words & valid_words

    #update log probabilities based on word likelihoods from the common words
    for word in common_words:
        positive_word_likelihood = positive_likelihoods[word]
        negative_word_likelihood = negative_likelihoods[word]

        positive_log_probability += math.log(positive_word_likelihood)
        negative_log_probability += math.log(negative_word_likelihood)

    # make prediction based on higher probability 
    if positive_log_probability > negative_log_probability:
        predicted_sentiment = 'positive'
    else:
        predicted_sentiment = 'negative'

    return predicted_sentiment

#TASK 6 
def task_six(training_data, training_labels, test_data, test_labels, k=5):
    print(f"\nRUNNING TASK SIX - performing K-Folds cross-validation to find optimal word length\n")
    optimal_word_length = task_six_part_a(training_data, training_labels, k)
    
    print(f"\nRUNNING TASK SIX - FINAL OUTPUT CALCULATING FOR OPTIMAL WORD LENGTH {optimal_word_length}...\n")
    task_six_part_b(training_data, training_labels, test_data, test_labels, optimal_word_length)

# Perform k-fold cross-validation to find the optimal word length
def task_six_part_a(training_data, training_labels,k):
    word_lengths = range(1, 11)
    mean_accuracies_results = []

    #convert data to numpy for efficiency
    training_data_array = np.array(training_data)
    training_labels_array = np.array(training_labels)

    #for each word length perform k-fold cross-validation
    for word_length in word_lengths:
        accuracies = []
        kf = KFold(n_splits=k, shuffle=True,random_state=42)
        # go through each fold
        for train_index, val_index in kf.split(training_data_array):
            #split data into training and evaluation sets
            reviews_train, reviews_val = training_data_array[train_index], training_data_array[val_index]
            sentiments_train, sentiments__val = training_labels_array[train_index], training_labels_array[val_index]

            #train the classifer on the training set and current word length
            positive_likelihoods, negative_likelihoods, positive_prior, negative_prior, valid_words = train_classifier(
                reviews_train, sentiments_train, word_length
            )

            # use the evaluation set to get predictions
            predictions = []
            for review_text in reviews_val:
                predicted_sentiment = task_five(
                    review_text,
                    positive_likelihoods,
                    negative_likelihoods,
                    positive_prior,
                    negative_prior,
                    valid_words
                )
                predictions.append(predicted_sentiment)

            # Calculate and store the accuracy for the current fold
            accuracy = accuracy_score(sentiments__val, predictions)
            accuracies.append(accuracy)

        #push the mean accuracy for the current word length
        mean_accuracy = np.mean(accuracies)
        mean_accuracies_results.append(mean_accuracy)
        print(f"Word Length: {word_length}, Mean Accuracy: {mean_accuracy:.4f}")

    #get the max accuracy and index of optimal word length.
    max_accuracy = max(mean_accuracies_results)
    optimal_index = mean_accuracies_results.index(max_accuracy)

    #return the optimal word length
    optimal_word_length = word_lengths[optimal_index]
    print(f"\nOptimal Word Length: {optimal_word_length}")
    return optimal_word_length

# use optimal word length and print final output values
def task_six_part_b(training_data, training_labels, test_data, test_labels, optimal_word_length):
    # train the classifer using the optimal word length and training set from task 1
    positive_likelihoods, negative_likelihoods, positive_prior, negative_prior, valid_words = train_classifier(
        training_data, training_labels, optimal_word_length
    )

    # use the evaluation set to get predictions
    predictions = []
    for review_text in test_data:
        predicted_sentiment = task_five(
            review_text,
            positive_likelihoods,
            negative_likelihoods,
            positive_prior,
            negative_prior,
            valid_words
        )
        predictions.append(predicted_sentiment)

    accuracy = accuracy_score(test_labels, predictions)
    c_matrix = confusion_matrix(test_labels, predictions, labels=['positive', 'negative'])

    # flatten the confusion matrix into a 1D array.
    true_positive, false_negative, false_positive, true_negative = c_matrix.ravel()
    total = true_negative + true_positive + false_negative + false_positive 
    true_negative_rate = true_negative / total * 100
    true_positive_rate = true_positive / total * 100
    false_negative_rate = false_negative / total * 100
    false_positive_rate = false_positive / total * 100

    print("\nFinal Evaluation....")

    print("Confusion Matrix:")
    print(c_matrix)

    print(f"True Negative Percentages: {true_negative} ({true_negative_rate:.4f}%)")
    print(f"\nTrue Positive Percentages: {true_positive} ({true_positive_rate:.4f}%)")
    print(f"False Negative Percentages: {false_negative} ({false_negative_rate:.4f}%)")
    print(f"False Positive Percentages: {false_positive} ({false_positive_rate:.4f}%)")

    print(f"\nClassification Accuracy Score: {accuracy:.4f}")

# this calls the tasks 2-4 in the same way as the main function but the data is specified on the parameter training data and word length.
def train_classifier(reviews_train, sentiments_train, word_length):
    #TRAINING BASED ON TASKS 2 - 4
    #TASK 2 USAGE - clean and filter words by parameters.
    cleaned_training_data, filtered_words = task_two(reviews_train, word_length, word_occurrences=6)
    #set for performance
    valid_words = set(filtered_words)

    #TASK 3 USAGE
    #run task three to get mappings of frequency for each word in positive and negative arrays.
    positive_word_map, negative_word_map = task_three(cleaned_training_data, sentiments_train, valid_words)
    
    #TASK 4 USAGE
    #convert np array to list and count postives / negatives
    positive_reviews_training_count = list(sentiments_train).count('positive')
    negative_reviews_training_count = list(sentiments_train).count('negative')

    positive_likelihoods, negative_likelihoods, positive_prior, negative_prior = task_four(
        positive_reviews_training_count,
        negative_reviews_training_count,
        positive_word_map,
        negative_word_map,
        valid_words
    )

    return positive_likelihoods, negative_likelihoods, positive_prior, negative_prior, valid_words

#OUTPUT
if __name__ == "__main__":
  print("\n=== RUNNING THE ASSIGNMENT ===\n")

  print("\nRUNNING TASK ONE - Seperate data into meaningful categories for testing and evaulation\n")
  #TASK 1 OUTPUTS
  training_data, test_data, training_labels, test_labels, positive_reviews_training_count, negative_reviews_training_count = task_one()

  # TASK 2 OUTPUTS 
  print("\nRUNNING TASK TWO - Clean data and produce word-list based on defined parameters of word size and word occurrance amount. \n")
  # clean and filter words by parameters.
  cleaned_training_data, filtered_words = task_two(training_data, word_length=5, word_occurrences=6)
  #convert to set for performance
  valid_words = set(filtered_words)
  #TASK 2 RELATED INFORMATION FOR CONTEXT IN CONSOLE
  print(f"Total Amount of cleaned reviews: {len(cleaned_training_data)}")
  print(f"Sample of cleaned words: {list(valid_words)[:10]}")

  # TASK 3 OUTPUTS 
  print("\nRUNNING TASK THREE -  Mapping frequenices of times a word appears in the different reviews for both positve and negative\n")
  #run task three to get mappings of frequency for each word in positive and negative arrays.
  positive_word_map, negative_word_map = task_three(cleaned_training_data, training_labels, valid_words)
  #TASK 3 RELATED INFORMATION FOR CONTEXT IN CONSOLE
  print(f"Number of words in positive word map: {len(positive_word_map)}")
  print(f"Number of words in negative word map: {len(negative_word_map)}")
  print("Sample of word frequencies in positive reviews:")
  for word in list(positive_word_map)[:5]:
      print(f"'{word}': {positive_word_map[word]}")
  print("Sample of word frequencies in negative reviews:")
  for word in list(negative_word_map)[:5]:
     print(f"'{word}': {negative_word_map[word]}")
  print("These mappings show how often each word appears in positive versus negative reviews.\n")

  # TASK 4 OUTPUTS
  print("\nRUNNING TASK FOUR - Getting likelihoods and prior values for positive / negative words and reviews.\n")
  positive_likelihoods, negative_likelihoods, positive_prior, negative_prior = task_four(
      positive_reviews_training_count, negative_reviews_training_count, positive_word_map, negative_word_map, valid_words
  )
  #TASK 4 RELATED INFORMATION FOR CONTEXT IN CONSOLE
  print(f"Prior probability of a positive review: {positive_prior:.4f}")
  print(f"Prior probability of a negative review: {negative_prior:.4f}")
  
  sample_words = list(valid_words)[:5]
  print("\nSample word likelihoods:")
  for word in sample_words:
    print(f"Word: '{word}' | P(word|positive): {positive_likelihoods[word]:.4f} | P(word|negative): {negative_likelihoods[word]:.4f}")
  print("These values show the estimated likelihood of a word appearing in a review given its sentiment.\n")

  #TASK 5 OUTPUT
  new_review_text = "the FILM was GOOD, GREAT, FANTASTIC, I liked it"
  print(f"\nRUNNING TASK FIVE - Classify a new review using Naive Bayes, for the review...\n '{new_review_text}' \n")
  predicted_sentiment = task_five(
      new_review_text,
      positive_likelihoods,
      negative_likelihoods,
      positive_prior,
      negative_prior,
      valid_words
  )
  #TASK 5 RELATED INFORMATION FOR CONTEXT IN CONSOLE
  print(f"The predicted sentiment for the review is: {predicted_sentiment.upper()}")

  #TASK 6 OUTPUT
  task_six(training_data, training_labels, test_data, test_labels)