import copy
import os
import shutil
from inspect import signature
import json


# Importing various model classes
from network.models.add_logits_model import AddLogitsModel
from network.models.add_scaled_logits_model import AddScaledLogitsModel
from network.models.is_in_max_wire_model import IsInMaxWireModel
from network.models.is_in_relation import IsInRelationModel
from network.models.lstm_model import LSTMModel
from network.models.textspace_model import TextspaceModel
from network.models.weighted_sum_of_wires_one_network import \
    WeightedSumOfWiresModel
from network.trainers.individual_networks_trainer import \
    IndividualNetworksTrainer
from network.trainers.one_network_trainer import OneNetworkTrainer

# Set environment variable to disable GPU (assuming CUDA is available)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import necessary libraries
import pickle
from datetime import datetime
from pathlib import Path

from tensorflow import keras
import wandb
from wandb.integration.keras import WandbCallback

# Import custom callbacks and functions
from utils.callbacks import ValidationAccuracy, \
    ModelCheckpointWithoutSaveTraces
from sklearn.model_selection import train_test_split

# Define the base path to the project directory.
# You may need to modify this path according to your project structure.
# this should the path to \Neural-DisCoCirc
base_path = os.path.abspath("/Users/vincent/Documents/GitHub/neural_discocirc")

# Configuration dictionaries for output and training.
output_config = {
    "log_wandb": False,
    "wandb_project": "discocirc",
    "tb_callback": False,
    "save_model": True,
    "run_test_dataset": False,
}

training_config = {
    "batch_size": 32,
    "dataset_size": -1,  # -1 for entire dataset
    "epochs": 20,
    "learning_rate": 0.01,
    "model": AddLogitsModel,
    "task": 1,
    "trainer": OneNetworkTrainer,
    # "trainer": IndividualNetworksTrainer,
    "dataset_split": ("random", 1), # (split_type, random state)
    # "dataset_split": ("depth", [1]) # (split_type, depths of training set)

}

model_configs = {
    "wire_dimension": 10,
    "hidden_layers": [10, 10],
    "is_in_hidden_layers": [10, 10],
    "relevance_hidden_layers": [10, 10],
    "softmax_relevancies": False,
    "softmax_logits": False,
    "expansion_hidden_layers": [20, 50],
    "contraction_hidden_layers": [50, 20],
    "latent_dimension": 100,
    "textspace_dimension": 20,
    "qna_hidden_layers": [10, 10],
    "lstm_dimension": 10,
}

# Function to split the dataset based on the depth of questions.
def train_test_depth_split(dataset, training_depths):
    split_datasets = []
    previous_length = 0
    counter = 0
    for q in dataset:
        if len(q['context_circ']) < previous_length:
            counter = 0
        previous_length = len(q['context_circ'])
        if len(split_datasets) <= counter:
            split_datasets.append([])
        split_datasets[counter].append(q)
        counter += 1

    training_dataset = []
    validation_dataset = []
    for i, set in enumerate(split_datasets):
        if i in training_depths:
            training_dataset += split_datasets[i]
        else:
            validation_dataset += split_datasets[i]

    return training_dataset, validation_dataset

# Function to train the neural network.
def train(base_path, save_path, vocab_path,
          data_path):
    # Create model_config dictionary based on model_class constructor parameters.
    print('Create model_config...')
    model_class = training_config['model']
    training_method = training_config['trainer']
    model_config = {}
    for val in signature(model_class.__init__).parameters:
        if val not in model_configs.keys():
            continue
        model_config[val] = model_configs[val]
    print('Configs created.')

    # Construct the path for saving the model.
    print('Creating save directory')
    # Construct the full path for saving the model, ensuring it's valid and writable
    save_base_path = os.path.join(base_path, save_path, model_class.__name__)
    os.makedirs(save_base_path, exist_ok=True)  # Creates the directory if it doesn't exist
    # Ensure the directory is writable
    if not os.access(save_base_path, os.W_OK):
        raise ValueError(f"The directory {save_base_path} is not writable.")
    print('Save directory created.')

    # Update training_config with model_config and other parameters.
    training_config.update(model_config)
    training_config['hidden_layers'] = model_configs['hidden_layers']
    if output_config["log_wandb"]:
        # Initialize Wandb (if logging is enabled).
        print("Initialise wandb...")
        wandb.init(project=output_config["wandb_project"], entity="domlee",
                   config=training_config)

    train_dataset_name = "task{:02d}_train.p".format(training_config["task"])

    print('Training: {} with trainer {} on data {}'
          .format(model_class.__name__,
                  training_config['trainer'].__name__,
                  train_dataset_name))

    # Load vocabulary.
    vocab_file = base_path + vocab_path + "task{:02d}_train.p".format(training_config["task"])
    print('Loading vocabulary: {}'.format(vocab_file))
    with open(vocab_file, 'rb') as file:
        lexicon = pickle.load(file)
        print(f"lexicon first element is of type: {type(lexicon[0])} !!!")
    dataset_file = base_path + data_path + train_dataset_name
    print('Loading pickled dataset: {}'.format(dataset_file))
    with open(dataset_file, "rb") as f:
        # Load dataset which is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)
        if training_config['dataset_size'] != -1:
            dataset = dataset[:training_config['dataset_size']]

    print('Splitting dataset...')
    if training_config['dataset_split'][0] == 'random':
        # Random split of dataset.
        train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=training_config['dataset_split'][1])
    elif training_config['dataset_split'][0] == 'depth':
        # Split based on question depth.
        train_dataset, validation_dataset = train_test_depth_split(dataset,
                                                            training_depths=training_config['dataset_split'][1])
    
    '''
    # Debugging: Print the intended save path before trainer initialization
    print(f"Intended model_save_path: {save_base_path}")

    # Check if the path is a string and points to a valid directory
    if isinstance(save_base_path, str) and os.path.isdir(save_base_path):
        print(f"Confirmed: {save_base_path} is a valid directory.")
    else:
        print(f"Error: {save_base_path} is not a valid directory.")

    # Check if the directory is writable
    if os.access(save_base_path, os.W_OK):
        print(f"Confirmed: {save_base_path} is writable.")
    else:
        print(f"Error: {save_base_path} is not writable.")
    '''

    print('Initializing trainer...')
    discocirc_trainer = training_config['trainer'](lexicon=lexicon,
                            model_class=model_class,
                            hidden_layers=training_config['hidden_layers'],
                            question_length = len(dataset[0]['question']),
                            model_save_path=save_base_path,  # Pass the correct, writable directory path here
                            **model_config
    )

    # Compile the model.
    discocirc_trainer.model_class.build([])
    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=training_config["learning_rate"]),
        run_eagerly=True
    )

    # Initialize callback for validation accuracy.
    datetime_string = datetime.now().strftime("%B_%d_%H_%M")
    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                    interval=1, log_wandb=output_config["log_wandb"])

    print('Training...')

    callbacks = [validation_callback]

    # Optionally, add TensorBoard and ModelCheckpoint callbacks.
    if output_config['tb_callback']:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir='logs/{}'.format(datetime_string),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq='batch',
        )
        callbacks.append(tb_callback)

    if output_config['save_model']:
        checkpoint_callback = ModelCheckpointWithoutSaveTraces(
            filepath='{}/{}'.format(save_base_path, datetime_string),
            save_freq=20 * training_config["batch_size"]
        )
        callbacks.append(checkpoint_callback)

    if output_config["log_wandb"]:
        callbacks.append(WandbCallback())

    # Fit the model.
    discocirc_trainer.fit(
        train_dataset,
        validation_dataset,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks
    )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)
    print("The accuracy on the train set is", accuracy)

    if output_config["log_wandb"]:
        wandb.log({"train_accuracy": accuracy})

    if output_config["run_test_dataset"]:
        print("Getting the test accuracy")
        test_dataset_name = "task{:02d}_test.p".format(training_config["task"])
        with open(base_path + data_path + test_dataset_name, 'rb') as f:
            test_dataset = pickle.load(f)

        test_accuracy = discocirc_trainer.get_accuracy(
            discocirc_trainer.compile_dataset(test_dataset))
        print("The accuracy on the test set is", test_accuracy)

        if output_config["log_wandb"]:
            wandb.log({"test_accuracy": test_accuracy})

    if output_config['save_model']:
        # Generate a unique name for the save directory using the current UTC timestamp.
        name = os.path.join(save_base_path, model_class.__name__ + "_" + datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S"))
        
        # The directory for the specific save instance is created above...
        # ...save_model method inherited from baseclass saves weights there
        discocirc_trainer.save_model(name)

        # additionally, create and save a JSON file for model_config
        model_config_path = os.path.join(name, "model_configs.json")
        with open(model_config_path, 'w') as model_config_file:
            json.dump(model_configs, model_config_file, indent=4)
        
        # Uncommenting the following line causes size issues with the zip protocol.
        #shutil.make_archive(name, 'zip', save_base_path)
        
        # If logging with wandb is enabled, save the zip file to wandb.
        if output_config["log_wandb"]:
            wandb.save(name + '.zip')

# Entry point of the script.
if __name__ == "__main__":
# Generate a unique name for the save directory using the current UTC timestamp.
    train(base_path,
          "saved_models/",
          "/data/task_vocab_dicts/",
          "/data/pickled_dataset/")
