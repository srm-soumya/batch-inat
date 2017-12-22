import argparse
import glob
import os
import pickle
import time
import pandas as pd
import numpy as np
import cntk as C
import _cntk_py
import cntk.io.transforms as xforms
from collections import Counter
from PIL import Image
from pprint import pprint as pp
from cntk.train.training_session import *
from cntk.logging import *
from cntk.debugging import *

parser = argparse.ArgumentParser(description='Image Classification Model')
parser.add_argument('--preprocess', action='store_true', help='Preprocess and create the map files', default=False)
parser.add_argument('--train', action='store_true', help='Train the model', default=False)
parser.add_argument('--data_dir', '-d', action='store', help='Specify the data directory', default='data')
parser.add_argument('--metadata_dir', '-dd', action='store', help='Specify the metadata directory', default='metadata')
parser.add_argument('--model_dir', '-m', action='store', help='Specify the pretrained model directory', default='model')
parser.add_argument('--output_dir', '-o', action='store', help='Specify the model directory', default='output')
parser.add_argument('--num_epochs', '-n', action='store', help='Number of Epochs to run', default=10, type=int)

parser.add_argument('--evaluate', action='store_true', help='Evaluate the model', default=False)
parser.add_argument('--model', action='store', help='Model Path')

args = parser.parse_args()

DATADIR = args.data_dir                        # DATADIR - holds the train, validation, test images
METADATADIR = args.metadata_dir                # METADATADIR - holds the input_map files
MODELDIR = args.model_dir                      # MODELDIR - holds the pre-trained model
OUTPUTDIR = args.output_dir                    # OUTPUTDIR - will store the checkpoint and trained-model
TRAINDIR = os.path.join(DATADIR, 'train')      # TRAINDIR - path to the training directory
TESTDIR = os.path.join(DATADIR, 'test')        # TESTDIR - path to the test directory
NUMEPOCHS = args.num_epochs                    # NUMEPOCHS - number of epochs you wish to train your model

model_name = 'resnet18-inat.model'

# id2label, label2id are mappers for label-index
id2label = dict()
label2id = dict()

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3
num_classes  = sum(1 for i in os.listdir(os.path.join(TRAINDIR)))
epoch_size = None

def create_map_file(directory):
    '''
    Input: name of the directory for which you need the map_file
    Function:
    - Checks for class imbalance in the data, tries to overcome it by creating duplicate entries
      for classes with low ratio. With Image Augmentation applied on duplicated data, the class
      imbalance issue is somewhat solved.
      (Credit for this: https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras)
    - Creates the respective map files and stores them in the metadata repository.
    Return: None
    '''
    df = pd.DataFrame([])
    df['path'] = list(glob.iglob(os.path.join(DATADIR, directory, '*', '*')))
    df['label'] = df['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    if not label2id:
        print('label2id mapper doesnot exists')
        labels = np.sort(df.label.unique())

        for index, label in enumerate(labels):
            label2id[label] = index
            id2label[index] = label

    # Add id to each image path
    df['id'] = df['label'].map(label2id)

    # Balance weights
    print('Before Balancing')
    print(df['id'].value_counts())

    # Computes the dictionary of weights for different classes of images in the dataset
    def get_class_weights(y, smooth_factor=0.15):
        counter = Counter(y)
        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p
        majority = max(counter.values())
        return {cls: int(majority // count) for cls, count in counter.items()}

    copy = df.copy()
    d = get_class_weights(df['id'].values)

    # Create copies of images based on their weights calculated, to balance the dataset
    for id, count in d.items():
        count -= 1
        if count:
            subset = df[df['id'] == id]
            copy = copy.append([subset] * count, ignore_index=True)

    df = copy
    print('After Balancing')
    print(df['id'].value_counts())

    df[['path', 'id']].to_csv(os.path.join(METADATADIR, '{}_map.tsv'.format(directory)), index=False, header=False, sep='\t')

def store_in_pickle(data, filename):
    ''' Takes a data structure and stores it with the given filename in metadata dir '''
    with open(os.path.join(METADATADIR, filename), 'wb') as file:
        pickle.dump(data, file)

def store_map_files():
    ''' Creates the map file for training and validation set '''
    create_map_file('train')
    create_map_file('validation')
    store_in_pickle(label2id, 'label2id')
    store_in_pickle(id2label, 'id2label')

def create_image_mb_source(map_file, train, total_number_of_samples):
    '''
    Input: map_file, train:bool, total_num_of_samples
    Function:
    - checks if it is training or testing phase
    - for training: It applies Image Augmentation techniques like cropping, width_shift, height_shift,
                    horizontal_flip, color_contrast to prevent overfitting of the model.
    Return: MinibatchSource to be fed into the CNTK Model
    '''
    print('Creating source for {}.'.format(map_file))
    transforms = []
    if train:
        # Apply translational and color transformations only for the Image set
        transforms += [
            xforms.crop(crop_type='randomarea', area_ratio=(0.08, 1.0), aspect_ratio=(0.75, 1), jitter_type='uniratio'), # train uses jitter
            xforms.color(brightness_radius=0.4, contrast_radius=0.4, saturation_radius=0.4)
        ]

    # Scale the images to a specified size (224 x 224) as expected by the model
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='cubic')
    ]

    return C.io.MinibatchSource(
        C.io.ImageDeserializer(
            map_file,
            C.io.StreamDefs(features=C.io.StreamDef(field='image', transforms=transforms), # 1st col in mapfile referred to as 'image'
                            labels=C.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=False)

def resnet_model(name, scaled_input):
    '''
    Input: pretrained-model name, scaled_input
    Function:
    - We are using Transfer Learning here, since the iNaturalist Image dataset is similar to Imagenet data.
    - Load Resnet34 as the base-model
    - Finetune Resnet34 by removing the last layer and add custom layers.
    - Custom layers:
        - Dense
        - Dropout
        - BatchNorm
    Return: Model
    '''
    print('Loading Resnet model from {}.'.format(name))
    base_model = C.load_model(os.path.join(MODELDIR, name))
    feature_node = C.logging.find_by_name(base_model, 'features')
    last_node = C.logging.find_by_name(base_model, 'z.x')

    # Clone the desired layers with fixed weights
    cloned_layers = C.combine([last_node.owner]).clone(C.CloneMethod.clone, {feature_node: C.placeholder(name='features')})
    cloned_out = cloned_layers(scaled_input)

    # Add GlobalPooling followed by a dropout layer
    z = C.layers.GlobalAveragePooling()(cloned_out)
    z = C.layers.Dropout(dropout_rate=0.3, name='d1')(z)

    # Add first block of dense layers
    z = C.layers.Dense(128, activation=C.ops.relu, name='fc1')(cloned_out)
    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Dropout(dropout_rate=0.6, name='d2')(z)

    # Add second block of dense layers
    z = C.layers.Dense(128, activation=C.ops.relu, name='fc2')(cloned_out)
    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Dropout(dropout_rate=0.3, name='d2')(z)

    z = C.layers.Dense(num_classes, activation=None, name='prediction')(z)

    return z

# TODO: Load the trained model and unfreeze the layers
# def load_and_unfreeze_model(name, scaled_input):
#     print('Loading trained model from {}.'.format(os.path.join(OUTPUTDIR, name)))
#     path = os.path.join(OUTPUTDIR, name)
#     if not os.path.exists(path):
#         raise FileNotFoundError('Initial model of phase 1 doesnot exist, please complete the initial steps!')
#     model = C.load_model(path)
#     clone = model.clone(C.CloneMethod.clone)
#     z = clone(scaled_input)
#     return z

def create_resnet_network():
    ''' Create the Resnet Network '''
    print('Creating the network.')
    # Input variables denoting the features and label data
    feature_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # Scale the input, by subtracting it from the mean of Imagenet dataset.
    scaled_input = feature_var - C.constant(114)
    z = resnet_model('ResNet18_ImageNet_CNTK.model', scaled_input)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    C.logging.log_number_of_parameters(z)

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

def create_trainer(network, epoch_size, num_quantization_bits, warm_up, progress_writers):
    ''' Create Trainer '''
    print('Creating the trainer.')
    # Differential Learning rate scheduler
    lr_schedule = C.learning_rate_schedule([0.01] * 10 + [0.001] * 20 + [0.0001] * 30, unit=C.UnitType.minibatch)
    mm_schedule = C.momentum_schedule(0.9)
    l2_reg_weight = 0.0001

    # Create the Adam learner
    learner = C.adam(network['output'].parameters,
                  lr_schedule,
                  mm_schedule,
                  l2_regularization_weight=l2_reg_weight,
                  unit_gain=False)

    # Compute the number of workers
    num_workers = C.distributed.Communicator.num_workers()
    print('Number of workers: {}'.format(num_workers))

    if num_workers > 1:
        parameter_learner = C.train.distributed.data_parallel_distributed_learner(learner, num_quantization_bits=num_quantization_bits)
        trainer = C.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_writers)
    else:
        trainer = C.Trainer(network['output'], (network['ce'], network['pe']), learner, progress_writers)

    return trainer

def train_model(network, trainer, train_source, test_source, validation_source, minibatch_size,
    epoch_size, restore, profiling=False):
    ''' Train the model '''
    print('Training the model.')
    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Enable profiling
    if profiling:
        start_profiler(sync_gpu=True)

    # Callback functionn called after each epoch
    def callback(index, average_error, cv_num_samples, cv_num_minibatches):
        print('Epoch:{}, Validation Error: {}'.format(index, average_error))
        return True

    # Create multiple checkpoints for your model, trainer and minibatches so that you can recover in case of failure.
    checkpoint_config = CheckpointConfig(frequency=epoch_size * 5, filename=os.path.join(OUTPUTDIR, "resnet34_cp"), restore=restore)
    # test_config = TestConfig(minibatch_source=test_source, minibatch_size=minibatch_size)
    validation_config = CrossValidationConfig(
        minibatch_source=validation_source,
        frequency=epoch_size,
        minibatch_size=minibatch_size,
        callback=callback)

    start = time.time()

    # Train the model
    training_session(
        trainer=trainer,
        mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=checkpoint_config,
        # test_config=test_config,
        cv_config=validation_config
    ).train()

    end = time.time()
    print('The Network took {} secs to train.'.format(end - start))
    print('Saving the model here {}.'.format(os.path.join(OUTPUTDIR, model_name)))
    # save the model
    trainer.model.save(os.path.join(OUTPUTDIR, model_name))

    if profiling:
        stop_profiler()

def run(train_data, test_data, validation_data, minibatch_size=200, epoch_size=50000,
    num_quantization_bits=32, warm_up=0, num_epochs=100, restore=True, log_to_file='logs.txt',
    num_mbs_per_log=100, profiling=True):
    _cntk_py.set_computation_network_trace_level(0)

    # Create the network to be trained
    network = create_resnet_network()

    # Define the ProgessWriter
    progress_writers = [C.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=os.path.join(OUTPUTDIR, log_to_file),
        rank=C.train.distributed.Communicator.rank(),
        num_epochs=num_epochs,
        distributed_freq=None)]

    # Create the trainer
    trainer = create_trainer(network, epoch_size, num_quantization_bits, warm_up, progress_writers)

    # Create the input data sources
    train_source = create_image_mb_source(train_data, train=True, total_number_of_samples=num_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    validation_source = create_image_mb_source(validation_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)

    # Call the train_model function
    train_model(network, trainer, train_source, validation_source, test_source, minibatch_size, epoch_size, restore, profiling)

def predict_image(model, image_path):
    '''
    Input: model, image_path
    Function: Passes the Image through the network, computes the output.
    Return: softmax output of the image
    '''
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    try:
        img = Image.open(image_path)
        # Resize the image
        resized = img.resize((image_width, image_height), Image.ANTIALIAS)
        # RGB => BGR
        bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
        # CHW => HWC
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {model.arguments[0]: [hwc_format]}
        output = model.eval(arguments)

        # Compute Softmax function over the output
        sm = C.softmax(output[0])

        return sm.eval()
    except Exception as e:
        print(e)
        print("Could not open (skipping file): {}".format(image_path))
        return None

def evaluate_model(model):
    '''
    Goes through the training and test set, computes the confusion-matrix, top_1, top_5% score
    for each class of the labels for both categories and sub-categories.
    '''
    # train_dict and test_dict stores the top 1% and top 5% score for each class
    train_dict = {}
    test_dict = {}

    # confusion_matrix stores sub-category level info, cat_confusion_matrix stores category level info
    confusion_matrix = {}
    cat_confusion_matrix = {}

    for label in os.listdir(TRAINDIR):
        cat = label.split('-')[0]
        print(cat)

        print('Processing {}'.format(label))
        train_dict[label] = {
            'top_1': 0,
            'top_5': 0,
            'total': 0
        }

        id = label2id[label]

        for image in os.listdir(os.path.join(TRAINDIR, label)):
            prediction = predict_image(model, os.path.join(TRAINDIR, label, image))
            if prediction is not None:
                train_dict[label]['total'] += 1
                predicted_label = id2label[np.argmax(prediction)]
                if id == np.argmax(prediction):
                    train_dict[label]['top_1'] += 1
                if id in np.argsort(prediction)[-5:]:
                    train_dict[label]['top_5'] += 1

    for label in os.listdir(TESTDIR):
        cat = label.split('-')[0]
        print(cat)

        print('Processing {}'.format(label))
        test_dict[label] = {
            'top_1': 0,
            'top_5': 0,
            'total': 0
        }

        cat_confusion_matrix[cat] = {}
        confusion_matrix[label] = {}

        for other_label in os.listdir(TESTDIR):
            other_cat = other_label.split('-')[0]
            cat_confusion_matrix[cat][other_cat] = 0
            confusion_matrix[label][other_label] = 0

        id = label2id[label]

        for image in os.listdir(os.path.join(TESTDIR, label)):
            prediction = predict_image(model, os.path.join(TESTDIR, label, image))
            if prediction is not None:
                test_dict[label]['total'] += 1
                predicted_label = id2label[np.argmax(prediction)]
                pred_cat = predicted_label.split('-')[0]
                cat_confusion_matrix[cat][pred_cat] += 1
                confusion_matrix[label][predicted_label] += 1
                if id == np.argmax(prediction):
                    test_dict[label]['top_1'] += 1
                if id in np.argsort(prediction)[-5:]:
                    test_dict[label]['top_5'] += 1

    ccfm = pd.DataFrame(cat_confusion_matrix)
    cfm = pd.DataFrame(confusion_matrix)

    train_df = pd.DataFrame(train_dict).transpose()
    train_df['top_1_%'] = (train_df['top_1'] / train_df['total']) * 100
    train_df['top_5_%'] = (train_df['top_5'] / train_df['total']) * 100

    test_df = pd.DataFrame(test_dict).transpose()
    test_df['top_1_%'] = (test_df['top_1'] / test_df['total']) * 100
    test_df['top_5_%'] = (test_df['top_5'] / test_df['total']) * 100

    writer = pd.ExcelWriter('evaluation-matrix.xlsx')
    train_df.to_excel(writer, 'Train')
    test_df.to_excel(writer, 'Test')
    ccfm.to_excel(writer, 'cat_confusion_matrix')
    cfm.to_excel(writer, 'confusion-matrix')
    writer.save()
    print('\nDone.\n')

if __name__=='__main__':
    if args.preprocess:
        print('\nStarting Preprocessing...\n')
        store_map_files()

    # Load id2label and label2id dicts
    id2label = pickle.load(open(os.path.join(METADATADIR, 'id2label'), 'rb'))
    label2id = pickle.load(open(os.path.join(METADATADIR, 'label2id'), 'rb'))

    # Count the number of images in the training set
    epoch_size = sum(1 for line in open(os.path.join(METADATADIR, 'train_map.tsv')))
    print('Number of classes: {}'.format(num_classes))
    print('Epoch size: {}'.format(epoch_size))

    # Load train, test and validation data
    train_data = os.path.join(METADATADIR, 'train_map.tsv')
    test_data = os.path.join(METADATADIR, 'validation_map.tsv')
    validation_data = os.path.join(METADATADIR, 'validation_map.tsv')

    if args.train:
        print('\nStarting Training...\n')
        run(train_data, test_data, validation_data, epoch_size=epoch_size, num_epochs=NUMEPOCHS)

    if args.evaluate:
        print('\nStarting Evaluation...\n')
        model = C.load_model(args.model)
        evaluate_model(model)

    # Must call MPI finalize when process exit without exceptions
    C.train.distributed.Communicator.finalize()
