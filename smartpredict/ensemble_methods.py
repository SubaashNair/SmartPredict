# smartpredict/ensemble_methods.py
"""
Ensemble methods module for SmartPredict.
Provides functions to create ensemble models.
"""

from sklearn.ensemble import StackingClassifier

def apply_ensembles(base_learners, meta_learner):
    """
    Apply ensemble methods to combine base learners with a meta learner.

    Parameters:
    base_learners (list of tuples): List of base learner models.
    meta_learner (estimator): The meta learner model.

    Returns:
    estimator: The stacked ensemble model.
    """
    ensemble_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
    return ensemble_model