class Metric:
    """
    A metric class that encapsulates:
    - The computation function
    - Whether it needs probabilities (True) or class predictions (False)
    - Whether higher values are better (True) or worse (False)
    """

    def __init__(self, function, needs_probabilities: bool, higher_is_better: bool):
        self.function = function
        self.needs_probabilities = needs_probabilities
        self.higher_is_better = higher_is_better

    def __call__(self, y_true, y_pred_or_prob):
        return self.function(y_true, y_pred_or_prob)
