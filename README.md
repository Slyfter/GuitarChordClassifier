# Guitar Chord Image Classifier
This repository contains a machine learning project to build a guitar chord image classifier using the ResNet152 model and fine-tune it on open guitar chords.

## Project Goal/Motivation
The motivation behind this project is to address a common challenge faced by guitar learners: recognizing and identifying chords played by other musicians when learning a new song. Many beginners grapple with this aspect of learning, which can slow their progress and dampen their enjoyment of the instrument. Moreover, most of the existing solutions to this problem are audio-based, requiring a clear sound file of the chord being played. This presents limitations in certain scenarios - for example, if a learner is trying to capture a chord being played during a live concert, the audio quality may be poor due to the extreme volume and subpar microphone quality in many smartphones. A snapshot of the musician in action could provide a clearer reference, but there are few tools available to analyze such images. The aim of this project, therefore, is to develop an accurate and efficient image classification system that can classify images of open guitar chords into different classes. With this tool, guitar learners will be able to swiftly and accurately identify various chords from images - for instance, a photo taken during a concert - thereby enhancing their overall musical proficiency and enjoyment.

## Data Collection & Generation
The data used for this project is composed of images of various guitar chords, gathered from friends (mostly through Instagram, WhatsApp, and face-to-face), internet data sets (e.g. https://www.kaggle.com/datasets/michaeltenoyo/fretboardnet), and frames extracted from YouTube guitar tutorials (e.g. https://www.youtube.com/watch?v=qAlyjGrThGo&pp=ygUdZ3VpdGFyIGNob3JkcyBmaW5nZXIgcG9zaXRpb24%3D) using the `frameextraction.py` script included in the repository. This wide range of sources ensures a robust and diverse dataset as open chords can be played in different styles.

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

I have included a Jupyter Notebook named `fine_tuning_resnet.ipynb` in the repository. This notebook provides detailed markdown comments covering the steps and explanations for the **Interpretation and Validation** of the fine-tuned ResNet model. By referring to this notebook, you can gain insights into the interpretation of results and the validation process employed in this project.

## Results
After training, the model's performance is evaluated on a test set. The performance of the model, in terms of accuracy, is then printed. The model's parameters which produced the lowest validation loss during training are saved and can be used later for further evaluation or deployment.

## Reproducing this Project using Google Colab
Note: The model has been trained using my local GPU. In case, you do not have access to a GPU in your local machine, consider using Google Colab. Here's a guide on how to set-up google colab: 

1. Google Colab Setup: Start by going to the Google Colab website. You'll need to sign in with your Google account. After signing in, click on File > Upload notebook and upload the `fine_tuning_resnet.ipynb` file.

2. Access to GPU: Once your notebook is uploaded, you need to change the runtime type to use a GPU. Click on Runtime (Laufzeit) > Change runtime type (Laufzeittyp ändern) and select GPU from the Hardware accelerator drop-down list.

3. Importing Project Files: To access other project files (like the dataset), you can use Google Drive. Use the following command to mount your google drive: 

 ```from google.colab import drive ```
 ```drive.mount('/content/drive')```

 After mounting the drive, upload your dataset to your Google Drive and you can access it from the colab notebook. If your file path in Google Drive is /content/drive/My Drive/Data/, use that file path to load the data into the variables `train_dir`, `test_dir`, and `unknown_dir`.

4. Running Cells: Now you can run each cell in the notebook by clicking on it and pressing Shift+Enter. Be sure to run them in order as they may depend on previous cells.

5. If you encounter any library-related issues (note that Google Colab usually comes with all the required libraries pre-installed), you can resolve them by running the  following command to install all the necessary libraries:
 ```!pip install -r requirements.txt```