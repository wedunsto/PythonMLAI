# Import necessary libraries
from sklearn.linear_model import LinearRegression
from PreProcessing import PreProcessor

class LinearRegressionTraining:
    def __init__(self):
        self.preProcessor = PreProcessor()
        self.predictedDependentVariables = None

        """Build and train Linear Regression model"""
        # Build Linear Regression model
        self.linearRegression = LinearRegression()

    def trainLinearRegressionModel(self):
        """Pre-process the dataset"""
        self.preProcessor.preProcess()

        # Train Linear Regression model to the training set
        self.linearRegression.fit(self.preProcessor.trainingFeatures, self.preProcessor.trainingDependentVariables)

        """Predict existing values using trained Linear Regression model"""
        self.predictedDependentVariables = self.linearRegression.predict(self.preProcessor.testingFeatures)
