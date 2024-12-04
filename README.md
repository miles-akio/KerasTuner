# KerasTuner

```markdown
# KerasTuner: Hyperparameter Tuning for MNIST Classification

This project demonstrates the use of **KerasTuner** for hyperparameter tuning in a neural network built with TensorFlow/Keras. It uses the MNIST dataset, a classic dataset for handwritten digit classification, to optimize the network's architecture and training parameters. 

## Features

- **Hyperparameter Tuning**: Uses KerasTuner's `RandomSearch` to experiment with:
  - Number of hidden layers
  - Number of units in each layer
  - Activation functions
  - Learning rate and optimizer hyperparameters (`beta_1`, `beta_2`)
- **Customizable Model Building**: A function dynamically builds models based on hyperparameters selected during tuning.
- **Dataset Preprocessing**: Includes data normalization, reshaping, and splitting into training, validation, and test sets.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required Python libraries:
   ```bash
   pip install tensorflow keras keras-tuner scikit-learn matplotlib
   ```

---

## Code Walkthrough

### 1. Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
```

### 2. Load and Preprocess MNIST Dataset
- **Normalization**: Pixel values scaled between 0 and 1.
- **Reshaping**: Inputs reshaped to 4D tensors for compatibility with convolutional layers.
- **Splitting**: Training data split into sub-training and validation sets.
```python
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

sub_train, valid, y_sub_train, y_valid = train_test_split(x_train, y_train, test_size=10000, random_state=42)

sub_train = sub_train.reshape((50000, 28, 28, 1))
valid = valid.reshape((10000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
```

### 3. Define the Model Builder
Defines the neural network structure, including:
- Variable number of layers (`num_layers`)
- Units per layer and activation functions
- Learning rate and optimizer parameters
```python
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Flatten(input_shape=(28, 28, 1)))
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'tanh', 'sigmoid'])
        ))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG'),
            beta_1=hp.Float('beta_1', min_value=0.8, max_value=0.99, step=0.01),
            beta_2=hp.Float('beta_2', min_value=0.9, max_value=0.999, step=0.01)
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 4. Instantiate the KerasTuner RandomSearch
```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='my_dir',
    project_name='mnist_hyperparameter_tuning'
)
```

### 5. Find the Best Hyperparameters
```python
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
```

### 6. Train the Best Model
```python
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(sub_train, y_sub_train, validation_data=(valid, y_valid), epochs=10, batch_size=128)
```

---

## Results

- The model achieves high accuracy on the MNIST dataset by dynamically tuning hyperparameters.
- The final architecture and optimizer settings are determined by KerasTuner's search process.

---

## Visualizing Results

Use Matplotlib to plot training and validation metrics:
```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## Directory Structure

```
my_dir/
├── mnist_hyperparameter_tuning/
│   ├── tuner0.json  # Saved tuner state
│   ├── trials/      # Contains trial results
```

---

## Notes

- Ensure TensorFlow and Keras are compatible with KerasTuner.
- Hyperparameter tuning can be computationally expensive; consider using GPUs for faster training.

---

## Future Improvements

- Experiment with convolutional layers to further enhance model performance.
- Add support for Bayesian Optimization or Hyperband in KerasTuner.
- Extend this project to other datasets or tasks.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```

This README.md file covers all aspects of the project, from setup to implementation and results. Let me know if you need further refinements!
