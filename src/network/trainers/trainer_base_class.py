from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type
from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf
import os

import json

"""
Subclasses need to concretely implement the following methods:

compile_diagrams()
"""

class TrainerBaseClass(keras.Model):
    def __init__(self,
                 wire_dimension: int,
                 lexicon: Dict[str, Any],
                 hidden_layers: List[int],
                 model_class: Type[keras.Model],
                 model_save_path: str,
                 **kwargs: Any
                 ):
        """Initializes the TrainerBaseClass with the given parameters.

        Args:
            wire_dimension: The dimension of the wire or embedding size.
            lexicon: A dictionary representing the lexicon or vocabulary.
            hidden_layers: A list of integers representing the size of hidden layers.
            model_class: The class of the model to be trained.
            model_save_path: The file system path where the model should be saved.
            **kwargs: Additional keyword arguments to pass to the model class initializer.
        """
        super().__init__()
        # Ensure model_save_path is a string
        if not model_save_path or not isinstance(model_save_path, str):
            raise ValueError("A valid model save path must be specified.")
        # Ensure model_save_path is a valid and writable directory
        if not os.path.isdir(model_save_path) or not os.access(model_save_path, os.W_OK):
            raise ValueError("The specified model save path is not a valid or writable directory.")
        
        # Initialize model parameters and the model itself based on provided class and parameters
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.lexicon = lexicon
        self.model_class = model_class(wire_dimension=wire_dimension,
                                       lexicon=lexicon, **kwargs)
        # Tracker for monitoring the loss during training
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Preprocesses and compiles the dataset for training.

        Args:
            dataset: A list of dictionaries, where each dictionary represents a data sample.

        Returns:
            A dictionary with keys corresponding to context, question, and answer, and values
            being the compiled data for each.
        """
        compiled_data = {}
        for key in [self.model_class.context_key,
                    self.model_class.question_key,
                    self.model_class.answer_key]:
            current_data = [data[key] for data in dataset]
            #Catching the tokens passed from ModelBaseClass
            if key in self.model_class.data_requiring_compilation:
                compiled_data[key] = self.compile_diagrams(current_data)
            else:
                compiled_data[key] = current_data

        return compiled_data

    # @tf.function
    def train_step_for_sample(self, batch_index: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs a training step for a given sample.

        Args:
            batch_index: A list of indices for the current batch.

        Returns:
            A tuple containing the loss and gradients for the batch.
        """
        # Perform a training step for a given sample, including gradient computation
        contexts = [self.dataset[self.model_class.context_key][i]
                    for i in batch_index]
        questions = [self.dataset[self.model_class.question_key][i]
                    for i in batch_index]
        answers = [self.dataset[self.model_class.answer_key][i]
                    for i in batch_index]
        with tf.GradientTape() as tape:
            context_output, question_output, answer_output = \
                self.call_on_dataset({
                    self.model_class.context_key: contexts,
                    self.model_class.question_key: questions,
                    self.model_class.answer_key: answers
                })
            loss = self.model_class.compute_loss(
                context_output, question_output, answer_output)
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    def get_config(self):
        # Return model configuration for serialization
        return {
            "wire_dimension": self.wire_dimension,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_config(cls, config):
        # Reconstruct the model from its configuration
        return cls(**config)

    @abstractmethod
    def load_model_trainer(model):
        # Abstract method to be implemented by subclasses for model loading
        pass

    @classmethod
    def load_model(cls, path, model_class):
        # Load a model from a given path, ensuring custom objects are recognized
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls,
                            model_class.__name__: model_class},
        )
        model.run_eagerly = True
        model = cls.load_model_trainer(model)
        return model

    def call_on_dataset(self, dataset):
        # Process the dataset through the model, handling data requiring compilation
        called_data = {}
        for key in [self.model_class.context_key,
                              self.model_class.question_key,
                              self.model_class.answer_key]:
            if key in self.model_class.data_requiring_compilation:
                called_data[key] = self.call(dataset[key])
            else:
                called_data[key] = dataset[key]

        return called_data[self.model_class.context_key], called_data[self.model_class.question_key], called_data[self.model_class.answer_key]

    def get_accuracy(self, dataset: List[Dict[str, Any]]) -> float:
        """Calculates the model's accuracy on the given dataset.

        Args:
            dataset: A list of dictionaries representing the dataset.

        Returns:
            The accuracy of the model as a float.
        """
        location_predicted = []
        location_true = []
        for i in range(len(dataset[self.model_class.context_key])):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')
            contexts, questions, answers = self.call_on_dataset({
                self.model_class.context_key: [dataset[self.model_class.context_key][i]],
                self.model_class.question_key: [dataset[self.model_class.question_key][i]],
                self.model_class.answer_key: [dataset[self.model_class.answer_key][i]]
            })
            answer_prob = self.model_class.get_answer_prob(contexts,
                                                           questions)
            location_predicted.append(
                self.model_class.get_prediction_result(answer_prob[0])
            )
            location_true.append(
                self.model_class.get_expected_result(answers[0])
            )
        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy

    def fit(self, train_dataset, validation_dataset, epochs, batch_size=32, **kwargs):
        # Prepare datasets and perform the training process
        print('compiling train dataset (size: {})...'.format(len(train_dataset)))

        self.dataset = self.compile_dataset(train_dataset)

        print('compiling validation dataset (size: {})...'.format(len(validation_dataset)))
        self.validation_dataset = self.compile_dataset(validation_dataset)

        input_index_dataset = tf.data.Dataset.range(len(train_dataset))
        input_index_dataset = input_index_dataset.shuffle(len(train_dataset))
        input_index_dataset = input_index_dataset.batch(batch_size)

        return super().fit(input_index_dataset, epochs=epochs, **kwargs)

    def save_model(self, path: str = None) -> None:
        """
        Saves the trained model to the specified path or to the default path
        set during initialization.

        Args:
            path (str, optional): The file path or directory to save the model.
                                  If None, uses the path provided during initialization.
        """
        if path is None:
            path = self.model_save_path
        if not path:
            raise ValueError("Model save path is not specified.")

        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the model's weights
        self.save_weights(os.path.join(path, "weights"), save_format='tf')

        # Save the model configuration as JSON
        model_config = self.get_config()
        config_path = os.path.join(path, "model_config.json")
        with open(config_path, 'w') as config_file:
            json.dump(model_config, config_file)

        print(f"Model saved to {path}")