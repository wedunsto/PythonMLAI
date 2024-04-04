# Import necessary libraries
import matplotlib.pyplot as plt
from LinearRegressionTraining import LinearRegressionTraining

linearRegressionModel = LinearRegressionTraining()
preProcessor = linearRegressionModel.preProcessor

"""Build, Train, and Use Linear Regression Model To Predict Values"""
linearRegressionModel.trainLinearRegressionModel()

"""Visualizing The Training Set Results"""

# This will compare the actual salaries seen for the test input values, versus the predicted salaries seen for the test values
plt.scatter(preProcessor.trainingFeatures, preProcessor.trainingDependentVariables, color = 'red')
plt.plot(preProcessor.trainingFeatures, linearRegressionModel.linearRegression.predict(preProcessor.trainingFeatures), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""Visualizing The Test Set Results"""

# Keep the training set in the predicted as the line would be the same
plt.scatter(preProcessor.testingFeatures, preProcessor.testingDependentVariables, color = 'red')
plt.plot(preProcessor.trainingFeatures, linearRegressionModel.linearRegression.predict(preProcessor.trainingFeatures), color = 'blue')
plt.title('Salary vs. Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
