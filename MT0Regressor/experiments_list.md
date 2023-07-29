1. Flan-T5-base-encoder + Dropout + MLP(ReLU)
2. Flan-T5-base-encoder + Dropout + MLP(ReLU) + Sigmoid
3. MT0-base-encoder + Dropout + MLP(ReLU) + Sigmoid
    1. Eval MSE: 1107
    2. Eval K-tau: 0.354
4. MT0-base-encoder + Dropout + MLP(ReLU) + Dropout + Sigmoid
    1. Eval MSE: 1144
    2. Eval K-tau: 0.365
5. MT0-base-encoder + Dropout + MLP(Tanh) + Dropout + Sigmoid(3 epochs)
    1. Eval MSE: 1074
    2. Eval K-tau: 0.374
6. MT0-base-encoder + Dropout + MLP(Tanh) + Dropout + Sigmoid(3 epochs) (SQM prompt)
    1. Eval MSE: 1100
    2. Eval K-tau: 0.373
7. LoRA MT0-base-encoder + Dropout + MLP(ReLU) + Dropout + Sigmoid(2 epochs)
    1. Eval MSE: 780
    2. Eval K-tau: 0.183
8. LoRA MT0-base-encoder + Dropout + MLP(ReLU + less layers) + Dropout + Sigmoid
    1. Eval MSE: 790
    2. Eval K-tau: 0.131
9. LoRA MT0-base-encoder + Dropout + MLP(ReLU + more layers) + Dropout + Sigmoid
    1. Eval MSE: 779
    2. Eval K-tau: 0.14
10. LoRA MT0-base-encoder + Dropout + MLP(Tanh) + Dropout + Sigmoid(2 epochs)
    1. Eval MSE: 779
    2. Eval K-tau: 0.23
11. LoRA MT0-large-encoder + Dropout + MLP(ReLU) + Dropout + Sigmoid(2 epochs)
    1. Eval MSE: 794
    2. Eval K-tau: 0.218
12. LoRA MT0-large-encoder + Dropout + MLP(Tanh) + Dropout + Sigmoid(3 epochs)
    1. Eval MSE: 994
    2. Eval K-tau: 0.259
