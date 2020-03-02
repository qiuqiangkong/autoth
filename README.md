# Autoth: Automatic optimzie hyper parameters

Autoth is a Python toolbox to automatically optimize hyper parameters to maximize scores. For example, Autoth can optimize hyper parameters to maximize F1 score in a classification task. In practice, Autoth numerically calculate gradients of scores over hyper parameters. Then, the hyper parameters are updated according to the gradients iteratively. Please see [1] for details.

## Install
```
pip install autoth
```

## Example
```
python3 example.py
```

## Results
<pre>
------ Manually selected hyper parameters ------
Hyper parameters: [0.3, 0.3, 0.3]
Score: 0.5556

------ Automatic optimized hyper parameters ------
Optimizing hyper parameters ...
learning rate: 0.010, total epochs: 10
    Hyper parameters: [0.3, 0.31, 0.29], score: 0.5556
    Epoch: 0, Time: 0.0181 s
    Hyper parameters: [0.3, 0.3197, 0.2801], score: 0.5556
    Epoch: 1, Time: 0.0178 s
    Hyper parameters: [0.3, 0.3237, 0.2702], score: 0.5714
    Epoch: 2, Time: 0.0212 s
    Hyper parameters: [0.3, 0.3245, 0.263], score: 0.6099
    Epoch: 3, Time: 0.0144 s
    Hyper parameters: [0.3, 0.3232, 0.2548], score: 0.6099
    Epoch: 4, Time: 0.0142 s
    Hyper parameters: [0.3, 0.3204, 0.2464], score: 0.6099
    Epoch: 5, Time: 0.0151 s
    Hyper parameters: [0.3, 0.3164, 0.2382], score: 0.6099
    Epoch: 6, Time: 0.0159 s
    Hyper parameters: [0.3, 0.316, 0.2302], score: 0.5940
    Epoch: 7, Time: 0.0143 s
    Hyper parameters: [0.3, 0.318, 0.2226], score: 0.5940
    Epoch: 8, Time: 0.0150 s
    Hyper parameters: [0.3, 0.3186, 0.2152], score: 0.6099
    Epoch: 9, Time: 0.0177 s

Optimized hyper parameters: [0.3, 0.3186, 0.2152]
Score: 0.6099
</pre>

## Cite
[1] Kong, Qiuqiang, Yong Xu, Wenwu Wang, and Mark D. Plumbley. "Sound Event Detection of Weakly Labelled Data with CNN-Transformer and Automatic Threshold Optimization." arXiv preprint arXiv:1912.04761 (2019).