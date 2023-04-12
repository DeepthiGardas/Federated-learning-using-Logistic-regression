from types import new_class
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the parameters of a scikit-learn LogisticRegression model."""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Sets the parameters of a scikit-learnLogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But the server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # dataset has 2 classes
    n_features = 5  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros(( n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    


# Create two functions load_data_client1() and for load_data_client1() loading the initially created datasets.
# Read data for the client 1 
def load_data_client1() -> Dataset:
    data = pd.read_csv('/home/ba-gardasd/FL/LR/client1.csv')
    data.rename({"Unnamed: 0":"a"}, axis=1, inplace=True)
    data.drop(["a"], axis=1, inplace=True)
    data.reset_index(drop=True)
    X = data.drop(columns=['Irrigation'], axis=1)
    y = data['Irrigation']
    dfX = np.array(X)
    y =np.array(y)

    # Standardizing the features
    try:
        scaler = StandardScaler()
        x = scaler.fit_transform(dfX)
    except ZeroDivisionError:
        return 0
    #x = StandardScaler().fit_transform(dfX)
    #x /= (np.std(x, axis=0) + 1e-8)
   
    """ Select the 80% of the data as Training data and 20% as test data """
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

# Read data for the client 2
def load_data_client2() -> Dataset:
    data = pd.read_csv('/home/ba-gardasd/FL/LR/client2.csv')
    data.rename({"Unnamed: 0":"a"}, axis=1, inplace=True)
    data.drop(["a"], axis=1, inplace=True)
    data.reset_index(drop=True)
    X = data.drop(columns=['Irrigation'], axis=1)
    y = data['Irrigation']
    dfX = np.array(X)
    y =np.array(y)
   
    # Standardizing the features
    try:
        scaler = StandardScaler()
        x = scaler.fit_transform(dfX)
    except ZeroDivisionError:
        return 0
    #x = StandardScaler().fit_transform(dfX)
    #x /= (np.std(x, axis=0) + 1e-8)
   
    """ Select the 80% of the data as Training data and 20% as test data """
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y Datasets"""
    randon_gen = np.random.default_rng()
    perm = randon_gen.permutation(len(X))
    return X[perm], y[perm]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y Datasets into a variety of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

