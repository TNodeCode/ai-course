# Loss Functions

A loss function, also known as a cost function or objective function, is a mathematical function that quantifies the discrepancy between the predicted outputs of a machine learning model and the true values or labels associated with the training data. It measures how well the model is performing in terms of its ability to make accurate predictions.

The purpose of a loss function is to provide a quantitative measure of the model's performance, which can be used to guide the learning process and update the model's parameters during training. By minimizing the loss function, the model aims to improve its predictions and align them with the true values.

For different types of problems in machine learning we need different loss functions. The two most convenient problems in machine learning are regression and classification.

**Regression** is a type of supervised learning task where the goal is to predict a continuous or numerical output variable. In regression, the target variable can take on a wide range of values within a continuous range. The objective is to learn a mapping or relationship between the input features and the continuous output variable.

**Classification**, on the other hand, is a type of supervised learning task where the goal is to predict a categorical or discrete output variable. In classification, the target variable represents specific classes or categories that the input data points belong to. The objective is to learn a decision boundary or classification rule that separates the input data into distinct classes.

We will now introduce you to the two most used loss functions for each of these problems.

## Regression Loss Functions

### MAE Loss Function (also L1 loss function)

L1 loss, also known as Mean Absolute Error (MAE) loss, is a commonly used loss function in regression tasks. It measures the average absolute difference between the predicted values and the true values of a regression problem.

Here's how the L1 loss works step by step:

Given a set of predicted values (ŷ) and true values (y) for a regression problem, the L1 loss calculates the absolute difference between each predicted value and its corresponding true value.

The absolute differences are then summed up.

The sum of absolute differences is divided by the total number of samples (n), giving us the mean absolute difference.

The L1 loss is the final output, representing the average absolute difference between the predicted and true values.

Mathematically, the L1 loss can be expressed as:

L1 = (1/n) \* Σ|ŷ - y|

Where:

ŷ represents the predicted value.
y represents the true value.
Σ denotes the summation symbol.
n represents the total number of samples.
The L1 loss has several desirable properties. It treats all errors equally, regardless of their magnitude, making it less sensitive to outliers compared to squared error-based losses like MSE. It is a non-negative value, where a lower L1 loss indicates better model performance in minimizing the difference between predicted and true values.

The L1 loss is commonly used in various regression algorithms, such as linear regression and decision trees. It serves as an optimization objective during training, where the goal is to minimize the L1 loss by adjusting the model's parameters to improve the accuracy of the predictions.

In addition to its use as a loss function, L1 regularization, also known as Lasso regularization, incorporates the L1 loss as a regularization term to promote sparsity in feature selection and model simplification.

### MSE Loss Function (also L2 loss function)

MSE (Mean Squared Error) loss is a commonly used loss function in regression tasks. It measures the average squared difference between the predicted values and the true values of a regression problem.

Here's how the MSE loss works step by step:

Given a set of predicted values (ŷ) and true values (y) for a regression problem, the MSE loss calculates the squared difference between each predicted value and its corresponding true value.

The squared differences are then summed up.

The sum of squared differences is divided by the total number of samples (n), giving us the mean squared difference.

The MSE loss is the final output, representing the average squared difference between the predicted and true values.

Mathematically, the MSE loss can be expressed as:

MSE = (1/n) \* Σ(ŷ - y)^2

Where:

ŷ represents the predicted value.
y represents the true value.
Σ denotes the summation symbol.
n represents the total number of samples.
The MSE loss has several desirable properties. It penalizes larger errors more than smaller errors due to the squaring operation, making it sensitive to outliers. It is a non-negative value, where a lower MSE indicates better model performance in minimizing the difference between predicted and true values.

The MSE loss is commonly used in various regression algorithms, such as linear regression, neural networks, and decision trees. It serves as an optimization objective during training, where the goal is to minimize the MSE loss by adjusting the model's parameters to improve the accuracy of the predictions.

## Classification Loss Functions

### Binary Cross Entropy Loss Function

Binary Cross Entropy (BCE) loss, also known as log loss or logistic loss, is a commonly used loss function in binary classification problems. It measures the dissimilarity between predicted probabilities and true binary labels.

Here's how the BCE loss works step by step:

Given a set of predicted probabilities (ŷ) and true binary labels (y) for a binary classification problem, the BCE loss calculates the logarithm of the predicted probability for the correct class.

The logarithmically transformed probabilities are summed up.

The sum is multiplied by -1.

The BCE loss is the final output, representing the average logarithmic loss over the samples.

Mathematically, the BCE loss can be expressed as:

BCE = - (1/n) _ Σ(y _ log(ŷ) + (1 - y) \* log(1 - ŷ))

Where:

ŷ represents the predicted probability for the positive class (class 1).
y represents the true binary label (0 or 1).
Σ denotes the summation symbol.
n represents the total number of samples.
The BCE loss has several desirable properties. It encourages the predicted probabilities to be close to 1 for positive class instances and close to 0 for negative class instances. It penalizes more significant deviations from the true labels through the logarithmic transformation. The BCE loss is widely used in logistic regression and binary classification algorithms, as it provides a measure of the dissimilarity between predicted probabilities and true labels, which can be minimized during model training.

It's important to note that BCE loss is typically used for binary classification problems where there are only two classes. For multi-class classification tasks, Cross Entropy loss or Categorical Cross Entropy loss is typically used instead.

### Cross Entropy Loss Function

Cross Entropy Loss, also known as Categorical Cross Entropy or Softmax Cross Entropy, is a commonly used loss function in multi-class classification problems. It measures the dissimilarity between predicted class probabilities and true class labels.

Here's how the Cross Entropy Loss works step by step:

Given a set of predicted class probabilities (ŷ) and true class labels (y) for a multi-class classification problem, the Cross Entropy Loss calculates the logarithm of the predicted probability for the correct class for each sample.

The logarithmic transformed probabilities are summed up.

The sum is multiplied by -1.

The Cross Entropy Loss is the final output, representing the average logarithmic loss over the samples.

Mathematically, the Cross Entropy Loss can be expressed as:

CE = - (1/n) _ Σ(Σ(y _ log(ŷ)))

Where:

ŷ represents the predicted probability for each class.
y represents the true one-hot encoded class label (a vector of 0s with a value of 1 for the true class).
Σ denotes the summation symbol.
n represents the total number of samples.
The Cross Entropy Loss encourages the predicted probabilities to be close to 1 for the true class and close to 0 for the other classes. It penalizes more significant deviations from the true labels through the logarithmic transformation. The Cross Entropy Loss is widely used in neural networks and other multi-class classification algorithms as an optimization objective during training.

It's important to note that the Cross Entropy Loss assumes that the predicted probabilities follow the softmax function, which ensures that the predicted probabilities sum up to 1 for each sample, making them valid probability distributions.

Alternatively, if the true labels are not one-hot encoded but expressed as integers representing class indices, the Cross Entropy Loss can be computed using the predicted log probabilities directly, without applying the logarithm function separately.

Overall, the Cross Entropy Loss provides a measure of dissimilarity between predicted class probabilities and true class labels that can be minimized during model training to improve the accuracy of multi-class classification models.
