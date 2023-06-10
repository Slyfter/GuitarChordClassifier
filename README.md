# Guitar Chord Image Classifier
This repository contains a machine learning project to build a guitar chord image classifier using the ResNet152 model and fine-tune it on open guitar chords.

## Project Goal
The motivation behind this project is to make a common challenge faced by guitar learners easier. Many beginners struggle with recognizing and identifying chords played by other musicians when learning a new song. The aim of this project is to develop an accurate and efficient image classification system that can classify images of open guitar chords into different classes. By doing so, it will empower guitar learners to swiftly and accurately identify various chords, thereby enhancing their overall musical proficiency and enjoyment.

## Data
The data used for this project is composed of images of various guitar chords, gathered from friends (using social networks), internet data sets, and frames extracted from YouTube guitar tutorials. This wide range of sources ensures a robust and diverse dataset as open chords can be played in different styles.

## Model
A pre-trained ResNet152 model is fine-tuned on the gathered data to perform the task of chord classification. The last fully connected layer of the model is adjusted to have an output size of 14 (equal to the number of chord classes).

## Getting Started
### Prerequisites
- Python 3.10
- PyTorch
- torchvision

### Installation
- Clone this repository using the following command after navigating to the target directory:

```git clone https://github.com/your_username/Guitar-Chord-Classifier.git```

- Install necessary packages by running the following command:

```pip install -r requirements.txt```

## Running the Project
1. Make sure your data is stored in the appropriate directories ('data/training' for training, 'data/test' for validation, and 'data/unknown' for additional tests with images the model never saw).
2. Run the script using the following command:

```python guitar_chord_classifier.py```

## Results
After training, the model's performance is evaluated on a test set. The performance of the model, in terms of accuracy, is then printed. The model's parameters which produced the lowest validation loss during training are saved and can be used later for further evaluation or deployment.
