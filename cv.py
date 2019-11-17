from util import accuracy, has_func, get_metrics

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

    def _update_scores(self, scores, agg):
        for metric, val in scores.items():
            agg[metric] += val
        return

    def _aggregate_scores(self, agg, n):
        scores = {}
        for metric, val in agg.items():
            scores[metric] = val / n
        return scores

    def cross_validate(self, learner, x, labels):
        if not has_func(learner, "fit") or not has_func(learner, "predict"):
            raise ValueError("Learner doesn't have fit(x) or predict(x) functions implemented")
        train_agg = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f-1": 0.0}
        val_agg = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f-1": 0.0}
        train_scores, val_scores = [], []
        for fold in range(self.k):
            training, val = self._partition(x, fold)
            training_labels, val_labels = self._partition(labels, fold)
            learner.fit(training)
            training_predicted = learner.predict(training)
            val_predicted = learner.predict(val)

            acc, (p, r, f1) = accuracy(training_labels, training_predicted), get_metrics(training_labels, training_predicted)
            scores = {"accuracy": acc, "precision": p, "recall": r, "f-1": f1}
            train_scores.append(scores)
            self._update_scores(scores, train_agg)

            acc, (p, r, f1) = accuracy(val_labels, val_predicted), get_metrics(val_labels, val_predicted)
            scores = {"accuracy": acc, "precision": p, "recall": r, "f-1": f1}
            val_scores.append(scores)
            self._update_scores(scores, val_agg)
        return self._aggregate_scores(train_agg, self.k), self._aggregate_scores(val_agg, self.k), train_scores, val_scores

