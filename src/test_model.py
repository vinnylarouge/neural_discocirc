import os

import numpy as np
from pandas import DataFrame

import pickle
from tensorflow import keras
import tensorflow as tf

import network

import json
import inspect

# Import necessary modules (uncomment as needed)
# from network.big_network_models.add_scaled_logits_one_network import AddScaledLogitsOneNetworkTrainer
# from network.big_network_models.is_in_one_network import IsInOneNetworkTrainer
# from network.big_network_models.one_network_trainer_base import OneNetworkTrainer
from network.trainers.individual_networks_trainer import IndividualNetworksTrainer
from network.models.add_logits_model import AddLogitsModel

# Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define the base path to the project directory.
# You may need to modify this path according to your project structure.
# this should the the path to \Neural-DisCoCirc
base_path = os.path.abspath("/Users/vincent/Documents/GitHub/neural_discocirc")
# base_path = os.path.abspath('.')

# Configuration dictionary containing various parameters.
config = {
    "trainer": IndividualNetworksTrainer, # Trainer class for the neural network.
    "model_class": AddLogitsModel, # Model class for the network.
    "dataset": "task01_test.p", # Dataset file name.
    "vocab": "task01_test.p", # Vocabulary file name.
    "weights": "AddLogitsModel/AddLogitsModel_2024_02_08_05_34_30" #Weights path
}

# Function to create a DataFrame for answers from the neural network.
def create_answer_dataframe(discocirc_trainer, vocab_dict, dataset):
    df = DataFrame([],
                   columns=['answer', 'correct', 'person', 'person_wire_no'])
    for i, (context_circuit_model, test) in enumerate(
            discocirc_trainer.dataset):
        person, location = test

        # Get the answer probabilities from the neural network.
        answer_prob = discocirc_trainer.call((context_circuit_model, person))
        answer_id = np.argmax(answer_prob)

        # Get the given answer, correct answer name, and other information.
        given_answer = list(vocab_dict.keys())[
                           list(vocab_dict.values()).index(answer_id)],
        correct_answer_name = dataset[i][0][person].boxes[0].name

        print("answer: {}, correct: {}, person: {}, {}".format(
            given_answer, location, person, correct_answer_name))

        df.loc[len(df.index)] = [
            given_answer, location, person, correct_answer_name]
    # Save the answers to a CSV file.
    df.to_csv("answers.csv")

# Function to test the neural network model.
def test(base_path, model_path, vocab_path, test_path):
    weights_base_path = base_path + model_path + config["weights"]
    test_base_path = base_path + test_path + config["dataset"]

    print('Testing: {} with weights from path {} on data {}'
          .format(config["model_class"].__name__, weights_base_path, test_base_path))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["vocab"], 'rb') as file:
        lexicon = pickle.load(file)

    print('loading pickled dataset...')
    with open(test_base_path, "rb") as f:
        dataset = pickle.load(f)[:50]  # Load the dataset, limit to the first 5 items for testing.
    print('compiling dataset (size: {})...'.format(len(dataset)))

    # Load the model configuration from the saved JSON file.
    model_config_path = os.path.join(weights_base_path, "model_configs.json") # Construct the model configuration file path
    with open(model_config_path, 'r') as config_file:
        model_config = json.load(config_file)

    # Get the constructor parameters of the model class
    model_constructor_params = inspect.signature(config["model_class"].__init__).parameters

    # Filter out unexpected keyword arguments from model_config
    model_config_filtered = {key: value for key, value in model_config.items() if key in model_constructor_params}

    # Add any additional required arguments here if needed
    model_config_filtered["model_class"] = config["model_class"]
    model_config_filtered["model_save_path"] = weights_base_path
    model_config_filtered["hidden_layers"] = model_config["hidden_layers"]

    # Initialize the trainer with the filtered model configuration
    discocirc_trainer = config["trainer"](lexicon=lexicon, **model_config_filtered)

    # Load the weights from the saved checkpoint files.
    print('loading weights...')
    checkpoint = tf.train.Checkpoint(model=discocirc_trainer)
    checkpoint_path = os.path.join(weights_base_path, "checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    # Compile the model with an optimizer.
    print('compiling model with optimizer...')
    discocirc_trainer.compile(optimizer=keras.optimizers.Adam(),
                              run_eagerly=True)

    # Calculate and print the accuracy on the test set.
    print('calculating accuracy...')
    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.compile_dataset(dataset))

    print("The accuracy on the test set is", accuracy)

    # Uncomment the following line to create a DataFrame with answers.
    # create_answer_dataframe(discocirc_trainer, vocab_dict, dataset)

# Entry point of the script.
if __name__ == "__main__":
    test(base_path,
         "/saved_models/",
         '/data/task_vocab_dicts/',
         "/data/pickled_dataset/")