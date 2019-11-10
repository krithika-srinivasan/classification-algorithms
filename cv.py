from util import accuracy, has_func

class CrossValidation:
    def __init__(self, k):
        self.k = k
        return

    def _partition(self, x, fold):
        """
        x - vector to partition
        fold - current fold number
        """
        num_samples = len(x)
        start = (num_samples // self.k) * fold
        end = (num_samples // self.k) * (fold + 1)

        validation = x[start:end]
        training = []
        training.extend(x[:start])
        training.extend(x[end:])
        return training, validation

    def cross_validate(self, learner, x, labels):
        if not has_func(learner, "fit") or not has_func(learner, "predict"):
            raise ValueError("Learner doesn't have fit(x) or predict(x) functions implemented")
        train_scores, val_scores = [], []
        for fold in range(self.k):
            training, val = self._partition(x, fold)
            training_labels, val_labels = self._partition(labels, fold)
            learner.fit(training)
            training_predicted = learner.predict(training)
            val_predicted = learner.predict(val)
            train_scores.append(accuracy(training_labels, training_predicted))
            val_scores.append(accuracy(val_labels, val_predicted))
        return train_scores, val_scores

