# DevSoc Assignment - Neural Network Framework

## Objective
In this assignment, you will create a simple neural network framework from scratch using NumPy. Your task is to implement various classes that handle the core components of a neural network, including layers, activation functions, loss functions, an optimizer, and a model class to encapsulate everything. Finally, you will use your framework to train a neural network on the MNIST dataset and achieve at least 80% accuracy on the test dataset from Kaggle.

## Instructions
- Implement the framework as described below. Ensure you do this in a `.py` file and document your code by adding docstrings.
- Import your framework into a Kaggle notebook (instructions provided below).
- Create a model using your framework and train it on the MNIST dataset.
- Evaluate your model by making a sample submission on Kaggle.
- Download the Kaggle notebook and upload your framework Python file along with the notebook to a GitHub repository.
- Add a README file illustrating how to use your framework, including loading your model weights. Also, include the link to your Kaggle notebook in it.

### Using Custom .py Files in Kaggle Notebooks
1. Create a Kaggle dataset.
2. Create a new notebook and add your dataset to the notebook.
3. Add the following code snippet:
   ```python
   import sys
   sys.path.append('/kaggle/input/{your-dataset-name}')
   ```
You can now import your files into the Kaggle notebook.

## Components to Implement
1. **Linear Layer Class**:
   - Implements a fully connected layer.
   - Methods: `forward`, `backward`.

2. **ReLU Activation Class**:
   - Implements the ReLU activation function.
   - Methods: `forward`, `backward`.

3. **Sigmoid Activation Class**:
   - Implements the Sigmoid activation function.
   - Methods: `forward`, `backward`.

4. **Tanh Activation Class**:
   - Implements the Tanh activation function.
   - Methods: `forward`, `backward`.

5. **Softmax Activation Class**:
   - Implements the Softmax activation function.
   - Methods: `forward`, `backward`.

6. **Cross-Entropy Loss Class**:
   - Implements the cross-entropy loss function. You can use the fusion method described in the PDF as well. See how `nn.CrossEntropyLoss` in PyTorch works.
   - Methods: `forward`, `backward`.

7. **Mean Squared Error (MSE) Loss Class**:
   - Implements the MSE loss function.
   - Methods: `forward`, `backward`.

8. **SGD Optimizer Class**:
   - Implements the stochastic gradient descent optimizer.
   - Methods: `step`.

9. **Model Class**:
   - Wraps everything into a cohesive model.
   - Methods: `add_layer`, `compile`, `train`, `predict`, `evaluate`, `save`, and `load`.

## Usage Example
```python
import numpy as np

# Define a simple neural network using the framework
model = Model()
model.add_layer(Linear(784, 128))
model.add_layer(ReLU())
model.add_layer(Linear(128, 10))
model.add_layer(Softmax())

# Compile the model with loss and optimizer
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)

# Assume x_train, y_train, x_test, y_test are preprocessed and available
# Train the model
model.train(x_train, y_train, epochs=20, batch_size=64)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
```

Follow these instructions carefully to complete the assignment and ensure your implementation is well-documented and tested.