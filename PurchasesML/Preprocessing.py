"""
Objective: Practice the pre processing steps of machine learning 
"""

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PreProcessing:
    def __init__(self):
        """Load dataset"""
        dataset = pd.read_csv('Data.csv')

        # Make the matrix of features and vector of dependent variables
        features = dataset.iloc[:, :-1].values
        dependentVariables = dataset.iloc[:, -1].values

        """Replace missing data in the features with that columns average"""
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(features[:,1:3])  # Apply to all rows, and columns EXCLUSIVELY
        features[:, 1:3] = imputer.transform(features[:, 1:3])  # Transform the data which the imputer was fit to

        """Encode string feature (Example: Country names) columns into categorical column data"""
        # Perform OneHotEncoding on the string column data and keep the columns that wont be encoded
        columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
        features = np.array(columnTransformer.fit_transform(features))  # Transform the matrix of features

        # Apply LabelEncoder to the dependent variable (Example: Yes or No)
        labelEncoder = LabelEncoder()
        dependentVariables = labelEncoder.fit_transform(dependentVariables)


        """Split the data into training and testing sets"""
        # Create matrix of features and vector of dependent variables for training and testing set
        # Randomly put 80% of data into training set and 20% into testing set
        self.trainingFeatures, self.testingFeatures, self.trainingDependentVariables, self.testingDependentVariables = train_test_split(features, dependentVariables, test_size=0.2, random_state=1)


        """Perform feature scaling on the training features"""
        # Using standardization to fit and transform training set of features
        standardScaler = StandardScaler()
        self.trainingFeatures[:, 3:] = standardScaler.fit_transform(self.trainingFeatures[:, 3:])
        self.testingFeatures[:, 3:] = standardScaler.transform(self.testingFeatures[:, 3:])

    def printFeatures(self):
        print(self.trainingFeatures)
        print()
        print(self.testingFeatures)

preProcessing = PreProcessing()
preProcessing.printFeatures()
