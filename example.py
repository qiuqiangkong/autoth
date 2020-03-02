import numpy as np
from sklearn import metrics
from autoth.core import HyperParamsOptimizer


class ScoreCalculatorExample(object):
    def __init__(self):
        """An example of Score calculator. Used to calculate score (such as F1), 
        given prediction, target and hyper parameters. 
        """
        self.N = 10     # Number of samples
        self.classes_num = 3    # Number of classes
        random_state = np.random.RandomState(1234)

        # Simulate system prediction. Usually a probability. 
        self.prediction = random_state.uniform(0, 1, (self.N, self.classes_num))

        # Simulate ground truth.
        self.target = random_state.randint(0, 2, (self.N, self.classes_num))

    def __call__(self, params):
        """Use hyper parameters to threshold prediction to obtain output.
        Then, the scores are calculated between output and target.
        """
        thresholds = params
        output = np.zeros_like(self.prediction)

        # Threshold to output
        for n in range(self.N):
            for k in range(self.classes_num):
                if self.prediction[n, k] > thresholds[k]:
                    output[n, k] = 1

        # Calculate score
        score = metrics.f1_score(self.target, output, average='macro')

        return score


if __name__ == '__main__':
    """An example of using autoth.
    """

    # Users should define this class
    score_calculator = ScoreCalculatorExample()

    print('------ Manually selected hyper parameters ------')
    init_params = [0.3, 0.3, 0.3]   # Initialize hyper parameters

    # Calculate score with initial hyper parameters
    score = score_calculator(init_params)

    print('Hyper parameters: {}'.format(init_params))
    print('Score: {:.4f}'.format(score))
    print()

    print('------ Automatic optimized hyper parameters ------')

    # Hyper parameter optimizer. This class will optimize hyper parameters to
    # maximize score calculated from score_calculator.
    hyper_params_opt = HyperParamsOptimizer(score_calculator, learning_rate=1e-2, epochs=10)

    # Optimize hyper parameters
    (opt_score, opt_params) = hyper_params_opt.do_optimize(init_params=init_params)

    print()
    print('Optimized hyper parameters: {}'.format([round(param, 4) for param in opt_params]))
    print('Score: {:.4f}'.format(opt_score))