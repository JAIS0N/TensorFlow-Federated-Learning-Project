## Code Explanation

This section explains the main code blocks used in the Federated Learning notebook. The project uses TensorFlow Federated to train a handwritten digit classification model on the EMNIST dataset using the Federated Averaging algorithm.

---

## 1. Installing Required Libraries

```python
!pip install --quiet --upgrade tensorflow
!pip install --quiet --upgrade tensorflow-federated
```

These commands install the required libraries.

* `tensorflow` is used for building and training the machine learning model.
* `tensorflow-federated` is used to simulate federated learning.
* `--quiet` reduces installation output.
* `--upgrade` installs the latest available version.

This setup is mainly intended for Google Colab or Jupyter Notebook.

---

## 2. Loading TensorBoard

```python
%load_ext tensorboard
```

TensorBoard is used to visualize training results such as loss and accuracy.

The `%load_ext tensorboard` command loads TensorBoard inside the notebook environment.

This is a Jupyter magic command, not regular Python syntax.

---

## 3. Importing Required Libraries

```python
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)
```

This code imports all required libraries.

### Explanation

```python
import collections
```

The `collections` module is used for special dictionary types such as:

* `defaultdict`
* `OrderedDict`

These are useful for organizing client data and formatting model input.

```python
import numpy as np
```

NumPy is used for numerical operations, especially calculating average images for each digit class.

```python
import tensorflow as tf
```

TensorFlow is used to build the neural network model, define the loss function, define metrics, and write TensorBoard logs.

```python
import tensorflow_federated as tff
```

TensorFlow Federated is used to load federated datasets and train the model using federated learning.

```python
np.random.seed(0)
```

This sets the NumPy random seed so NumPy-based operations are more reproducible.

---

## 4. Loading the Federated EMNIST Dataset

```python
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
```

This loads the EMNIST dataset in federated format.

Unlike a normal dataset, the data is split by clients.

Each client represents a separate local dataset.

For example:

```text
Client 1 data
Client 2 data
Client 3 data
...
Client N data
```

The returned objects are:

```python
emnist_train
emnist_test
```

* `emnist_train` contains federated training data.
* `emnist_test` contains federated testing data.

This project mainly uses `emnist_train`.

---

## 5. Creating a Dataset for One Client

```python
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]
)
```

This creates a TensorFlow dataset for one specific client.

### Explanation

```python
emnist_train.client_ids[0]
```

This selects the first client from the training dataset.

```python
create_tf_dataset_for_client(...)
```

This function retrieves all local examples for that specific client.

So `example_dataset` contains only the first client’s handwritten digit data.

---

## 6. Reading One Example from a Client

```python
example_element = next(iter(example_dataset))
```

This retrieves one example from the selected client dataset.

### Explanation

```python
iter(example_dataset)
```

Creates an iterator over the dataset.

```python
next(...)
```

Gets the first element from the iterator.

The result is stored in:

```python
example_element
```

Each example usually contains:

```python
example_element["pixels"]
example_element["label"]
```

* `pixels` contains the image data.
* `label` contains the correct digit class.

---

## 7. Viewing the Label of One Example

```python
example_element["label"].numpy()
```

This displays the label of the selected example.

The label is originally stored as a TensorFlow tensor.

The `.numpy()` method converts it into a normal NumPy value.

For example, if the output is:

```text
7
```

that means the image represents the digit `7`.

---

## 8. Displaying One Handwritten Digit Image

```python
from matplotlib import pyplot as plt

plt.imshow(example_element["pixels"].numpy(), cmap="gray", aspect="equal")
plt.grid(False)
_ = plt.show()
```

This code displays the handwritten digit image.

### Explanation

```python
from matplotlib import pyplot as plt
```

Imports Matplotlib for plotting images.

```python
plt.imshow(example_element["pixels"].numpy(), cmap="gray", aspect="equal")
```

Displays the image.

* `example_element["pixels"].numpy()` converts the image tensor into a NumPy array.
* `cmap="gray"` displays the image in grayscale.
* `aspect="equal"` keeps the image dimensions proportional.

```python
plt.grid(False)
```

Removes grid lines from the image.

```python
_ = plt.show()
```

Displays the final image.

The underscore `_` is used because the return value is not needed.

---

## 9. Displaying Multiple Images from One Client

```python
figure = plt.figure(figsize=(20, 4))
j = 0

for example in example_dataset.take(40):
    plt.subplot(4, 10, j + 1)
    plt.imshow(example["pixels"].numpy(), cmap="gray", aspect="equal")
    plt.axis("off")
    j += 1
```

This code displays 40 handwritten digit images from the same client.

### Explanation

```python
figure = plt.figure(figsize=(20, 4))
```

Creates a large figure for displaying multiple images.

```python
for example in example_dataset.take(40):
```

Takes the first 40 examples from the client dataset.

```python
plt.subplot(4, 10, j + 1)
```

Creates a subplot grid with 4 rows and 10 columns.

Since:

```text
4 × 10 = 40
```

this allows 40 images to be displayed.

```python
plt.imshow(...)
```

Displays each digit image.

```python
plt.axis("off")
```

Removes axis labels and tick marks.

```python
j += 1
```

Moves to the next subplot position.

---

## 10. Visualizing Label Distribution Across Clients

```python
f = plt.figure(figsize=(12, 7))
f.suptitle("Label Counts for a Sample of Clients")

for i in range(6):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[i]
    )

    plot_data = collections.defaultdict(list)

    for example in client_dataset:
        label = example["label"].numpy()
        plot_data[label].append(label)

    plt.subplot(2, 3, i + 1)
    plt.title("Client {}".format(i))

    for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
```

This code shows the number of examples for each digit label across different clients.

### Explanation

```python
for i in range(6):
```

Loops through the first 6 clients.

```python
client_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[i]
)
```

Gets the local dataset for client `i`.

```python
plot_data = collections.defaultdict(list)
```

Creates a dictionary where each label has a list of values.

Example:

```python
plot_data[0] = [0, 0, 0]
plot_data[1] = [1, 1]
plot_data[7] = [7, 7, 7, 7]
```

```python
label = example["label"].numpy()
plot_data[label].append(label)
```

Extracts the label and stores it in the dictionary.

```python
plt.hist(...)
```

Creates a histogram showing how many examples each client has for each digit.

This is important because federated learning data is often non-independent and non-identically distributed.

That means each client may have a different data distribution.

For example:

* One client may have many examples of digit `1`.
* Another client may have more examples of digit `7`.
* Another client may have very few examples overall.

This difference makes federated learning more challenging than centralized learning.

---

## 11. Visualizing Mean Images for Each Client

```python
for i in range(6):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[i]
    )

    plot_data = collections.defaultdict(list)

    for example in client_dataset:
        plot_data[example["label"].numpy()].append(example["pixels"].numpy())

    f = plt.figure(i, figsize=(12, 5))
    f.suptitle("Client #{}'s Locally Trained Model".format(i))

    for j in range(10):
        mean_img = np.mean(plot_data[j], 0)
        plt.subplot(2, 5, j + 1)
        plt.imshow(mean_img.reshape((28, 28)))
        plt.axis("off")
```

This code calculates and displays the average image for each digit label for each client.

### Explanation

```python
plot_data[example["label"].numpy()].append(example["pixels"].numpy())
```

Groups images by their labels.

For example:

```python
plot_data[3]
```

contains all images labeled as digit `3`.

```python
mean_img = np.mean(plot_data[j], 0)
```

Calculates the average image for digit `j`.

This is done pixel by pixel.

For example, if a client has 20 images of digit `5`, the code calculates the average of those 20 images.

```python
plt.imshow(mean_img.reshape((28, 28)))
```

Displays the average image as a 28 by 28 image.

This helps show that different clients may write digits differently.

---

## 12. Defining Training Constants

```python
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20

SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
```

These constants control how the data is prepared and how training is performed.

### Explanation

```python
NUM_CLIENTS = 10
```

Uses 10 clients during federated training.

```python
NUM_EPOCHS = 5
```

Each client repeats its local dataset 5 times during local training.

```python
BATCH_SIZE = 20
```

Each local batch contains 20 examples.

```python
SHUFFLE_BUFFER = 100
```

Controls how many examples are used for shuffling.

```python
PREFETCH_BUFFER = 10
```

Allows TensorFlow to prepare future batches in advance for better performance.

---

## 13. Preprocessing Client Data

```python
def preprocess(dataset):

    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.reshape(element["pixels"], [-1, 784]),
            y=tf.reshape(element["label"], [-1, 1])
        )

    return dataset.repeat(NUM_EPOCHS).shuffle(
        SHUFFLE_BUFFER, seed=1
    ).batch(
        BATCH_SIZE
    ).map(
        batch_format_fn
    ).prefetch(
        PREFETCH_BUFFER
    )
```

This function prepares each client’s local dataset for training.

### Explanation

The raw image shape is:

```text
28 × 28
```

The model expects a flat vector, so each image is reshaped into:

```text
784
```

because:

```text
28 × 28 = 784
```

### Inner Function

```python
def batch_format_fn(element):
```

This function formats each batch into the structure expected by TensorFlow Federated.

```python
x=tf.reshape(element["pixels"], [-1, 784])
```

Reshapes image pixels into flat vectors.

If the batch size is 20, the final shape becomes:

```text
20 × 784
```

```python
y=tf.reshape(element["label"], [-1, 1])
```

Reshapes labels into a column format.

If the batch size is 20, the label shape becomes:

```text
20 × 1
```

```python
collections.OrderedDict(...)
```

Returns the data in a consistent order.

TensorFlow Federated expects the input format to match the model input specification.

### Dataset Pipeline

```python
dataset.repeat(NUM_EPOCHS)
```

Repeats the client dataset for multiple local epochs.

```python
.shuffle(SHUFFLE_BUFFER, seed=1)
```

Randomizes the order of examples.

```python
.batch(BATCH_SIZE)
```

Groups examples into batches.

```python
.map(batch_format_fn)
```

Applies reshaping and formatting to each batch.

```python
.prefetch(PREFETCH_BUFFER)
```

Prepares future batches while the current batch is being processed.

---

## 14. Creating Federated Training Data

```python
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]
```

This function creates a list of preprocessed datasets, one dataset per client.

### Explanation

```python
client_data.create_tf_dataset_for_client(x)
```

Creates a TensorFlow dataset for client `x`.

```python
preprocess(...)
```

Applies preprocessing to that client’s dataset.

The final output is a list:

```python
[
    client_1_dataset,
    client_2_dataset,
    client_3_dataset,
    ...
]
```

TensorFlow Federated uses this list as federated training data.

---

## 15. Preparing the Selected Clients

```python
preprocessed_example_dataset = preprocess(example_dataset)

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(
    emnist_train,
    sample_clients
)
```

This code prepares the actual federated training data.

### Explanation

```python
preprocessed_example_dataset = preprocess(example_dataset)
```

Preprocesses one client dataset.

This is later used to define the model input specification.

```python
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
```

Selects the first 10 clients.

```python
federated_train_data = make_federated_data(...)
```

Creates the final federated dataset used for training.

---

## 16. Creating the Keras Model

```python
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer="zeros"),
        tf.keras.layers.Softmax(),
    ])
```

This function creates a simple neural network model.

### Model Architecture

```text
Input Layer → Dense Layer → Softmax Layer
```

### Explanation

```python
tf.keras.layers.InputLayer(input_shape=(784,))
```

The model accepts a flattened image with 784 pixel values.

```python
tf.keras.layers.Dense(10, kernel_initializer="zeros")
```

Creates a dense layer with 10 output neurons.

There are 10 neurons because the model predicts digits from 0 to 9.

```python
tf.keras.layers.Softmax()
```

Converts raw model outputs into probabilities.

For example, the output could look like:

```text
[0.02, 0.01, 0.05, 0.80, 0.01, 0.02, 0.03, 0.01, 0.03, 0.02]
```

The highest probability is at index `3`, so the model predicts digit `3`.

---

## 17. Wrapping the Keras Model for TensorFlow Federated

```python
def model_fn():
    keras_model = create_keras_model()

    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
```

TensorFlow Federated cannot directly use a normal Keras model. The model must be wrapped in a TensorFlow Federated-compatible format.

### Explanation

```python
keras_model = create_keras_model()
```

Creates a new Keras model.

```python
tff.learning.models.from_keras_model(...)
```

Converts the Keras model into a TensorFlow Federated model.

```python
input_spec=preprocessed_example_dataset.element_spec
```

Tells TensorFlow Federated the shape and structure of the input data.

The input data has the form:

```python
OrderedDict(
    x=<image_batch>,
    y=<label_batch>
)
```

```python
loss=tf.keras.losses.SparseCategoricalCrossentropy()
```

Defines the loss function.

Sparse categorical crossentropy is used because the labels are integers, not one-hot encoded vectors.

Example integer label:

```text
3
```

Example one-hot label:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

Since this notebook uses integer labels, sparse categorical crossentropy is correct.

```python
metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
```

Tracks model accuracy during training.

---

## 18. Building the Federated Averaging Training Process

```python
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)
```

This builds the federated learning training process.

The algorithm used is Federated Averaging.

### Explanation

```python
tff.learning.algorithms.build_weighted_fed_avg(...)
```

Creates a federated training algorithm using weighted Federated Averaging.

Federated Averaging works like this:

```text
1. Server sends global model to clients.
2. Clients train locally.
3. Clients send model updates to server.
4. Server averages the updates.
5. Server updates the global model.
```

```python
client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
```

Defines the optimizer used on each client.

Each client uses stochastic gradient descent with a learning rate of `0.02`.

```python
server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
```

Defines the optimizer used by the server.

A server learning rate of `1.0` means the server fully applies the averaged client update.

---

## 19. Initializing the Training State

```python
train_state = training_process.initialize()
```

This initializes the server-side training state.

The training state contains:

* Global model weights
* Server optimizer state
* Other internal federated learning state

At this point, the model is initialized but not trained yet.

---

## 20. Running One Federated Training Round

```python
result = training_process.next(train_state, federated_train_data)

train_state = result.state
train_metrics = result.metrics
```

This runs one round of federated training.

### Explanation

```python
training_process.next(train_state, federated_train_data)
```

Executes one federated learning round.

During this round:

```text
1. The global model is sent to the selected clients.
2. Each client trains locally.
3. Client updates are sent back to the server.
4. The server averages the updates.
5. The global model is updated.
```

```python
train_state = result.state
```

Updates the training state with the new global model.

```python
train_metrics = result.metrics
```

Stores the training metrics from that round.

---

## 21. Running Multiple Federated Training Rounds

```python
NUM_ROUNDS = 11

for round_num in range(1, NUM_ROUNDS):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    train_metrics = result.metrics

    print("round {:2d}, metrics={}".format(round_num, train_metrics))
```

This trains the federated model for multiple rounds.

### Explanation

```python
NUM_ROUNDS = 11
```

Defines the number of training rounds.

```python
for round_num in range(1, NUM_ROUNDS):
```

Runs rounds from 1 to 10.

```python
result = training_process.next(train_state, federated_train_data)
```

Runs one federated training round.

```python
train_state = result.state
```

Updates the global model state.

```python
train_metrics = result.metrics
```

Stores the metrics for the current round.

```python
print(...)
```

Prints the training metrics.

The metrics usually include values such as:

* training loss
* sparse categorical accuracy

---

## 22. Creating a TensorBoard Log Directory

```python
logdir = "/tmp/logs/scalars/training/"

try:
    tf.io.gfile.rmtree(logdir)
except tf.errors.NotFoundError as e:
    pass
```

This code creates a clean logging setup for TensorBoard.

### Explanation

```python
logdir = "/tmp/logs/scalars/training/"
```

Defines where TensorBoard logs will be stored.

```python
tf.io.gfile.rmtree(logdir)
```

Deletes previous logs from the same directory.

```python
except tf.errors.NotFoundError as e:
    pass
```

If the directory does not exist, the error is ignored.

This prevents old TensorBoard results from mixing with new ones.

---

## 23. Creating a TensorBoard Summary Writer

```python
summary_writer = tf.summary.create_file_writer(logdir)
train_state = training_process.initialize()
```

This creates a writer that saves metrics for TensorBoard.

### Explanation

```python
summary_writer = tf.summary.create_file_writer(logdir)
```

Creates a TensorBoard summary writer.

```python
train_state = training_process.initialize()
```

Reinitializes the training state.

Important: this starts training from scratch again for TensorBoard logging.

It does not continue from the earlier training loop.

---

## 24. Logging Metrics During Training

```python
with summary_writer.as_default():
    for round_num in range(1, NUM_ROUNDS):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics

        for name, value in train_metrics["client_work"]["train"].items():
            tf.summary.scalar(name, value, step=round_num)
```

This code trains the model and writes metrics to TensorBoard.

### Explanation

```python
with summary_writer.as_default():
```

Sets the TensorBoard writer as the default writer.

```python
for round_num in range(1, NUM_ROUNDS):
```

Runs multiple federated training rounds.

```python
result = training_process.next(train_state, federated_train_data)
```

Runs one round of federated learning.

```python
train_state = result.state
```

Updates the global model state.

```python
train_metrics = result.metrics
```

Stores metrics for the current round.

```python
for name, value in train_metrics["client_work"]["train"].items():
```

Loops through the training metrics.

Common metrics include:

```text
loss
sparse_categorical_accuracy
```

```python
tf.summary.scalar(name, value, step=round_num)
```

Writes each metric value to TensorBoard.

---

## 25. Launching TensorBoard

```python
!ls {logdir}

%tensorboard --logdir {logdir} --port=0
```

This displays the log files and launches TensorBoard.

### Explanation

```python
!ls {logdir}
```

Lists the TensorBoard event files.

```python
%tensorboard --logdir {logdir} --port=0
```

Starts TensorBoard.

```python
--logdir {logdir}
```

Tells TensorBoard where to find the logs.

```python
--port=0
```

Automatically selects an available port.

---

## 26. Complete Training Flow

The full notebook follows this process:

```text
Install libraries
        ↓
Import TensorFlow and TensorFlow Federated
        ↓
Load federated EMNIST data
        ↓
Select individual clients
        ↓
Visualize local client examples
        ↓
Analyze client label distributions
        ↓
Preprocess client datasets
        ↓
Create a Keras model
        ↓
Wrap the model for TensorFlow Federated
        ↓
Build Federated Averaging process
        ↓
Initialize global model
        ↓
Train over multiple federated rounds
        ↓
Record loss and accuracy
        ↓
Visualize metrics using TensorBoard
```

---

## 27. Main Federated Learning Logic

The most important code block is:

```python
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

train_state = training_process.initialize()

for round_num in range(1, NUM_ROUNDS):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    train_metrics = result.metrics

    print("round {:2d}, metrics={}".format(round_num, train_metrics))
```

This is where federated training actually happens.

In simple terms:

```text
The server creates a global model.
The selected clients receive the model.
Each client trains the model locally.
The server averages the client updates.
The global model improves after each round.
```

This is the core idea of Federated Learning.

---

## 28. Key Takeaways from the Code

This notebook demonstrates:

* How to load a federated dataset.
* How to inspect data from individual clients.
* How to visualize client-level data differences.
* How to preprocess local client data.
* How to define a simple TensorFlow model.
* How to convert a Keras model into a TensorFlow Federated model.
* How to train using Federated Averaging.
* How to track and visualize training metrics using TensorBoard.

The most important concept is that the model learns from multiple clients without directly combining their raw data into one central dataset.
