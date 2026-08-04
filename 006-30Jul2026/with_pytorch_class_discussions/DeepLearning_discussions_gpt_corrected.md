Activation functions:

1. Sigmoid
    1.1. Commonly used for binary classification.
    1.2. Outputs values between 0 and 1 that can be interpreted as probabilities.
    1.3. For example, 0.7 means the model predicts the positive class with 70% probability and the negative class with 30% probability.
    1.4. It is often paired with Binary Cross Entropy loss.

2. Softmax
    2.1. Commonly used for multi-class classification.
    2.2. Converts raw output scores into a probability distribution over classes.
    2.3. The number of outputs typically equals the number of classes.
    2.4. It is often paired with Categorical Cross Entropy loss.

3. Optimizers
    3.1. Adam — widely used for many networks and often a good default.
    3.2. SGD — standard optimizer commonly used for convolutional neural networks.
    3.3. SGD with momentum — accelerates convergence and reduces oscillation.
    3.4. Adagrad — adapts learning rates and can be useful for sparse data.
    3.5. Adadelta — builds on Adagrad with more robust updates.
    3.6. RMSProp — often used for recurrent neural networks and non-stationary objectives.

Detailed explanation and study guide:

- Sigmoid
    - Use case: binary classification problems such as spam detection or medical diagnosis.
    - Behavior: compresses inputs into the range (0, 1) using the formula `1 / (1 + exp(-x))`.
    - Strengths: easy to interpret as probability and works well for final output layer of binary classifiers.
    - Weaknesses: can saturate for large positive or negative inputs, causing vanishing gradients during training.
    - Tip: prefer `sigmoid` only on output layer and use it with binary cross entropy loss.

- Softmax
    - Use case: multi-class classification tasks such as digit recognition, image classification, or language modeling.
    - Behavior: converts input logits into probabilities using `exp(x_i) / sum_j exp(x_j)`.
    - Strengths: normalizes outputs so they sum to 1, making predictions mutually exclusive.
    - Weaknesses: can be sensitive to very large or small logits, but numerically stable implementations use log-sum-exp.
    - Tip: use softmax with categorical cross entropy loss and ensure the number of output neurons matches the number of classes.

- Optimizers
    - Adam: combines momentum and adaptive learning rates. Good default for many tasks and typically converges faster than vanilla SGD.
    - SGD: simple gradient descent with a fixed learning rate. Works well with careful tuning and often generalizes well.
    - SGD with momentum: helps escape shallow local minima and smooths gradient updates, especially useful for large networks.
    - Adagrad: adjusts learning rate per parameter based on historical gradients. Useful for sparse features, but learning rate can decay too quickly.
    - Adadelta: improves on Adagrad by restricting the window of accumulated gradients, so learning rates do not shrink indefinitely.
    - RMSProp: keeps a moving average of squared gradients to adapt the learning rate; popular for recurrent networks and noisy gradients.

Study guide:

- Start by understanding the difference between activation functions and loss functions. Activation functions shape the output of neurons, while loss functions measure model error.
- Memorize common layer combinations:
    - Binary output: Sigmoid + Binary Cross Entropy.
    - Multi-class output: Softmax + Categorical Cross Entropy.
- Practice with simple neural networks in frameworks like PyTorch or TensorFlow to see how sigmoid and softmax behave on real logits.
- Compare optimizers on the same model and dataset to observe differences in convergence speed, stability, and final accuracy.
- Read reference material or tutorials on gradient descent, momentum, and adaptive learning rates to understand why each optimizer behaves differently.
- When studying, write short notes that include formulas, advantages, disadvantages, and typical use cases for each activation function and optimizer.
