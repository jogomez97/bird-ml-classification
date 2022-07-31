# Bird ML Classifer

The aim of this work is to study the ability to use machine learning algorithms generally applied to image recognition for the classification of sounds of 20 different species of birds in the Aiguamolls de l'Empordà natural park in Catalonia, Spain.

For this purpose, it was decided to use different pre-trained CNNs and a set of manually labelled audios, processed in the form of spectrograms, to obtain different models that are able to predict with a high degree of confidence which species of bird is emitting sounds.

Additionally, a model to recognize bird species through images is accomplished. With both audio and image based models, several methods are studied to increase the accuracy of the classification by combining (or fusing) image and sound.

## Dataset

The audio dataset is published and accessible at: [Western Mediterranean Wetlands Bird Dataset](https://zenodo.org/record/5093173).

## Set up the environment

1. Clone this repository into local
2. Download the dataset.
3. Create a virtual environment with `python3 -m venv venv` (python 3.8.x is strongly recommended)
4. Activate the virtual environment with `source venv/bin/activate`
5. Install dependencies with `pip install -r requirements.txt` (for M1 macOS, install `requirements-macos.txt`)

Make sure that you have installed [the software pre-requisits](https://www.tensorflow.org/install/gpu#software_requirements) to run Tensorflow on a GPU. Check the trusted CUDA versions [here](https://www.tensorflow.org/install/source#tested_build_configurations).

## Run

It's important to have the environment correctly installed and activated. To run a single model training, execute `train.py` script with the appropiate arguments

Example:

```
python src/train.py -p 'path_to_dataset' -m resnet50 -b 16
```

The script will create an output folder `models` where it will store logs and trained models.

For help run from the base folder:

```
python src/train.py -h
```

## Run multiple trainings

To run multiple trainings a `run.sh` example script is provided in `bin` folder. To execute it simply do:

```
cd bin/
sh run.sh
```

## Get model scores

This repository includes a `get_model_scores.py` script that allows to test the all the models trained with `run.sh` at once and get accuracy, recall, precision and F1-scores in a CSV file.

## List of bird species

- Acrocephalus arundinaceus
- Acrocephalus melanopogon
- Acrocephalus scirpaceus
- Alcedo atthis
- Anas strepera
- Anas platyrhynchos
- Ardea purpurea
- Botaurus stellaris
- Charadrius alexandrinus
- Ciconia ciconia
- Circus aeruginosus
- Coracias garrulus
- Dendrocopos minor
- Fulica atra
- Gallinula chloropus
- Himantopus himantopus
- Ixobrychus minutus
- Motacilla flava
- Porphyrio porphyrio
- Tachybaptus ruficollis
