import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score, 
    classification_report
)

class ModelAssessment:
    """
    A class to evaluate and provide metrics for a machine learning model.

    Attributes
    ----------
    model : object
        The machine learning model to be evaluated.
    X_test : array-like
        The input features for the test set.
    y_test : array-like
        The true labels for the test set.
    y_pred : array-like
        The predicted labels for the test set, computed using the model.

    Methods
    -------
    accuracy():
        Returns the accuracy of the model on the test set.
    precision():
        Returns the weighted precision of the model on the test set.
    recall():
        Returns the weighted recall of the model on the test set.
    f1():
        Returns the weighted F1 score of the model on the test set.
    confusion_matrix():
        Returns the confusion matrix of the model on the test set.
    roc_auc():
        Returns the ROC AUC score for binary classification.
    classification_report():
        Returns a detailed classification report.
    summary():
        Returns a summary of all metrics.
    """

    def __init__(self, model, X_test, y_test):
        """
        Initializes the ModelAssessment with the model and test data.

        Parameters
        ----------
        model : object
            The machine learning model to be evaluated.
        X_test : array-like
            The input features for the test set.
        y_test : array-like
            The true labels for the test set.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def accuracy(self):
        """
        Calculates the accuracy of the model.

        Returns
        -------
        float
            The accuracy score.
        """
        return accuracy_score(self.y_test, self.y_pred)

    def precision(self):
        """
        Calculates the weighted precision of the model.

        Returns
        -------
        float
            The weighted precision score.
        """
        return precision_score(self.y_test, self.y_pred, average='weighted')

    def recall(self):
        """
        Calculates the weighted recall of the model.

        Returns
        -------
        float
            The weighted recall score.
        """
        return recall_score(self.y_test, self.y_pred, average='weighted')

    def f1(self):
        """
        Calculates the weighted F1 score of the model.

        Returns
        -------
        float
            The weighted F1 score.
        """
        return f1_score(self.y_test, self.y_pred, average='weighted')

    def confusion_matrix(self):
        """
        Computes the confusion matrix of the model.

        Returns
        -------
        array-like
            The confusion matrix.
        """
        return confusion_matrix(self.y_test, self.y_pred)

    def roc_auc(self):
        """
        Calculates the ROC AUC score for binary classification.

        Returns
        -------
        float or str
            The ROC AUC score if applicable, otherwise a message indicating it is not available.
        """
        if len(np.unique(self.y_test)) == 2:  # binary classification
            return roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        else:
            return "ROC AUC is only available for binary classification"

    def classification_report(self):
        """
        Generates a detailed classification report.

        Returns
        -------
        str
            The classification report.
        """
        return classification_report(self.y_test, self.y_pred)

    def summary(self):
        """
        Provides a summary of all evaluation metrics.

        Returns
        -------
        dict
            A dictionary containing all evaluation metrics.
        """
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1(),
            "confusion_matrix": self.confusion_matrix().tolist(),
            "roc_auc": self.roc_auc(),
            "classification_report": self.classification_report()
        }
