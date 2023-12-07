from allennlp.training.metrics.metric import Metric

@Metric.register("set-f1")
class SetBasedF1Measure(Metric):
    def __init__(self, averaging: str = "micro") -> None:
        assert averaging in ['micro', 'macro']
        self.averaging = averaging
        self.num_correct = None
        self.num_expected = None
        self.num_predicted = None

    def __call__(self, batch_predictions, batch_gold):
        assert isinstance(batch_predictions, list)
        assert isinstance(batch_gold, list)
        for predictions, gold in zip(batch_predictions, batch_gold):
            self.compute_metrics(predictions, gold)

    def compute_metrics(self, predictions, gold):
        assert isinstance(predictions, set) or isinstance(predictions, list)
        assert isinstance(gold, set) or isinstance(gold, list)
        if self.num_correct is None:
            self.num_correct = []
        if self.num_expected is None:
            self.num_expected = []
        if self.num_predicted is None:
            self.num_predicted = []
        prediction_types = set(type(p) for p in predictions)
        gold_types = set(type(g) for g in gold)
        if len(prediction_types) > 1:
            print('warning, more than one prediction type')
        if len(gold_types) > 1:
            print('warning, more than one prediction type')
        if (len(prediction_types & gold_types) == 0) and (len(prediction_types) > 0) and (len(gold_types) > 0):
            print('warning, prediction and gold are of different types')
        correct = set(predictions) & set(gold)
        self.num_correct.append(len(set(correct)))
        self.num_expected.append(len(set(gold)))
        self.num_predicted.append(len(set(predictions)))
        assert len(self.num_correct) == len(self.num_expected)
        assert len(self.num_correct) == len(self.num_predicted)

    def get_metric(self, reset: bool = False):
        invalid = any(x is None for x in [self.num_correct, self.num_expected, self.num_predicted])
        if invalid:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        if self.averaging == 'micro':
            precision, recall, f1 = [], [], []
            for c, e, p in zip(self.num_correct, self.num_expected, self.num_predicted):
                precision.append(1 if p == 0 else c/p)
                recall.append(1 if e == 0 else c/e)
                f1.append(1 if (e+p) == 0 else 2*c/(e+p))
            # end for c, e, p in zip(self.num_correct, self.num_expected, self.num_predicted):
            def mean(l):
                assert len(l)>0
                return sum(l)/len(l)
            precision = mean(precision)
            recall = mean(recall)
            f1 = mean(f1)
        else:
            c = sum(self.num_correct)
            e = sum(self.num_expected)
            p = sum(self.num_predicted)
            precision = c/p
            recall = c/e
            f1 = 2*c/(e+p)
        # end if self.averaging == 'micro':
        if reset:
            self.reset()
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def reset(self) -> None:
        self.num_correct = None
        self.num_expected = None
        self.num_predicted = None
