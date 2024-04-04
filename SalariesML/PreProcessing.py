"""
Objective: Perform pre-processing on the dataset
"""

# Import necessary libraries
import pandas as pd

from sklearn.model_selection import train_test_split

class PreProcessor:
    def __init__(self):
        self.trainingFeatures = None
        self.trainingDependentVariables = None
        self.testingFeatures = None
        self.testingDependentVariables = None

    def preProcess(self):  
        """Load the dataset"""
        dataset = pd.read_csv('Salary_Data.csv')

        # Create matrix of features and vector of dependent variables
        features = dataset.iloc[:, :-1].values # First column
        dependentVariables = dataset.iloc[:, -1].values

        """
        No need to replace missing data
        No categorical data to encode
        """

        """Split data into training and test sets"""
        # 80% of data is randomly used for training, 20% randomly used for testing
        self.trainingFeatures, self.testingFeatures, self.trainingDependentVariables, self.testingDependentVariables = train_test_split(features, dependentVariables, 
                                                                                                                                        test_size=0.2, random_state=0)