import argparse
import json
import os
import shutil
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

from custom_dataset_loader_and_splitter import CustomDatasetLoaderAndSplitter
from data_generator import DataGenerator
from fcheadnet import FCHeadNet


def build_output_folder(clear_output) -> str:
    if clear_output and os.path.exists("models"):
        shutil.rmtree("models")

    if not os.path.exists("models"):
        os.mkdir("models")

    now = datetime.now().strftime("%y-%m-%dT%H%M")
    output_path = os.path.join("models", f"train-{now}")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)
    return output_path


def train_mobilenet(args, output_path):
    # load the images and get the splits
    class_names = list(os.listdir(args.dataset_path))
    data_loader = CustomDatasetLoaderAndSplitter(
        args.dataset_path,
        validation=args.validation_split,
        test=args.test_split,
        seed=args.seed,
    )

    if args.test_split > 0:
        train_x, train_y, val_x, val_y, test_x, test_y = data_loader.load_and_split()
    else:
        train_x, train_y, val_x, val_y = data_loader.load_and_split()

    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    val_y = LabelBinarizer().fit_transform(val_y)

    # tensorflow tries to allocate the whole dataset to the GPU
    # so we need to provide a DataGenerator to feed the data in batches
    train_gen = DataGenerator(train_x, train_y, args.batch_size)
    val_gen = DataGenerator(val_x, val_y, args.batch_size)

    # set up test data if needed
    if args.test_split > 0:
        test_y = LabelBinarizer().fit_transform(test_y)
        test_gen = DataGenerator(test_x, test_y, args.batch_size)

    # load the pre-trained network, ensuring the head FC layer sets are left off (include_top=False)
    print("[INFO] loading model...")
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # initialize the new head of the network, a set of FC layers
    # followed by a softmax classifier
    head_model = FCHeadNet.build(base_model, len(class_names), 256)

    # place the head FC model on top of the base model -- this will
    # become the actual model we will train
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they
    # will *not* be updated during the training process
    for layer in base_model.layers:
        layer.trainable = False
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    print("[INFO] compiling model...")
    opt = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the head of the network for a few epochs (all other
    # layers are frozen) -- this will allow the new FC layers to
    # start to become initialized with actual "learned" values
    # versus pure random
    # typical warmup are 10-30 epoch
    print("[INFO] training head...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        verbose=1,
    )

    model.save(os.path.join(output_path, "warmup.h5"))
    with open(os.path.join(output_path, "history_warmup.json"), "w") as outfile:
        json.dump(history.history, outfile)
        outfile.close()

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    for layer in base_model.layers[15:]:
        layer.trainable = True

    # for the changes to the model to take affect we need to recompile
    # the model, this time using SGD with a *very* small learning rate
    print("[INFO] re-compiling model...")
    opt = SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # checkpoint
    filepath = os.path.join(
        output_path, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
    )
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)
    callbacks_list = [checkpoint, early_stop_callback]

    # train the model again, this time fine-tuning *both* the final set
    # of CONV layers along with our set of FC layers
    print("[INFO] fine-tuning model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        verbose=1,
        callbacks=callbacks_list,
    )

    # save the model to disk
    print("[INFO] serializing model...")
    model.save(os.path.join(output_path, args.output_model))
    with open(os.path.join(output_path, "history_training.json"), "w") as outfile:
        json.dump(history.history, outfile)
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--dataset-path", type=str, help="Path to the dataset.", required=True
    )
    parser.add_argument(
        "-o",
        "--output-model",
        type=str,
        default="output.h5",
        help="Output name of the trained model.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for training the model.",
    )
    parser.add_argument(
        "--validation-split",
        type=int,
        default=0.2,
        help="Percentage of dataset to split for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=0,
        help="Percentage of dataset to split for test.",
    )
    parser.add_argument(
        "--clear-models",
        action="store_true",
        help="Remove all subfolders in output models/ folder. Warning, this is not reversible!",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="Seed used to randomly split the dataset.",
    )
    args = parser.parse_args()

    output_path = build_output_folder(args.clear_models)

    train_mobilenet(args, output_path)
