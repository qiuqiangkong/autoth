import numpy as np
import time
        

class HyperParamsOptimizer(object):
    def __init__(self, score_calculator, learning_rate=1e-2, epochs=100, 
        step=0.01, max_search=5):
        """Hyper parameters optimizer. Parameters are optimized using gradient
        descend methods by using the numerically calculated graident: 
        gradient: f(x + h) - f(x) / (h)

        Args:
          score_calculator: object. See ScoreCalculatorExample in example.py as 
              an example.
          learning_rate: float
          epochs: int
          step: float, equals h for calculating gradients
          max_search: int, if plateaued, then search for at most max_search times
        """
        
        self.score_calculator = score_calculator
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = Adam()
        self.optimizer.alpha = learning_rate
        self.step = step
        self.max_search = max_search

    def do_optimize(self, init_params):
        print('Optimizing hyper parameters ...')
        print('learning rate: {:.3f}, total epochs: {}'.format(
            self.learning_rate, self.epochs))

        params = init_params.copy()

        for i in range(self.epochs):
            t1 = time.time()
            (score, grads) = self.calculate_gradients(params)
            grads = [-e for e in grads]
            params = self.optimizer.GetNewParams(params, grads)
            print('    Hyper parameters: {}, score: {:.4f}'.format([round(param, 4) for param in params], score))
            print('    Epoch: {}, Time: {:.4f} s'.format(i, time.time() - t1))
        
        return score, params

    def calculate_gradients(self, params):
        """Calculate gradient of thresholds numerically.
        Args:
          y_true: (N, (optional)frames_num], classes_num)
          output: (N, (optional)[frames_num], classes_num)
          thresholds: (classes_num,), initial thresholds
          average: 'micro' | 'macro'
        Returns:
          grads: vector
        """
        score = self.score_calculator(params)
        step = self.step
        grads = []

        for k, param in enumerate(params):
            new_params = params.copy()
            cnt = 0
            while cnt < self.max_search:
                cnt += 1
                new_params[k] += self.step
                new_score = self.score_calculator(new_params)

                if new_score != score:
                    break

            grad = (new_score - score) / (step * cnt)
            grads.append(grad)

        return score, grads


class Base(object):
    def _reset_memory(self, memory):
        for i1 in range(len(memory)):
            memory[i1] = np.zeros(memory[i1].shape)


class Adam(Base):
    def __init__(self):
        self.ms = []
        self.vs = []
        self.alpha = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.iter = 0
        
    def GetNewParams(self, params, gparams):
        if not self.ms:
            for param in params:
                self.ms += [np.zeros_like(param)]
                self.vs += [np.zeros_like(param)]
          
        # fast adam, faster than origin adam
        self.iter += 1
        new_params = []
        alpha_t = self.alpha * np.sqrt(1 - np.power(self.beta2, self.iter)) / (1 - np.power(self.beta1, self.iter))
        for i1 in range(len(params)):
            self.ms[i1] = self.beta1 * self.ms[i1] + (1 - self.beta1) * gparams[i1]
            self.vs[i1] = self.beta2 * self.vs[i1] + (1 - self.beta2) * np.square(gparams[i1])
            new_params += [params[i1] - alpha_t * self.ms[i1] / (np.sqrt(self.vs[i1] + self.eps))]
            
        return new_params
        
    def reset(self):
        self._reset_memory(self.ms)
        self._reset_memory(self.vs)
        self.epoch = 1