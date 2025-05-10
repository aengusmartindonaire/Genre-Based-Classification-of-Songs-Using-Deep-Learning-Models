# Music Genre Classification Project

## Overview
This project implements a music genre classification system using various neural network architectures. The system processes audio files from the GTZAN dataset to classify music into different genres based on extracted audio features. The codebase includes data preprocessing, exploratory data analysis (EDA), and model training using Deep Neural Networks (DNN), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN).

## Dataset
The dataset used is the GTZAN genre collection was downloaded at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification and stored in `/content/drive/MyDrive/Data/genres_original`. It contains audio files (WAV format) organized by genre, with each file sampled at 22,050 Hz. The dataset is accessed via Google Drive integration in a Google Colab environment.

## Project Structure
- **Notebook**: `genre-based-classification-modelling.ipynb`
  - Contains code for:
    - Loading and preprocessing audio files using `librosa`.
    - Visualizing audio waveforms for EDA.
    - Building, training, and evaluating DNN, CNN (with and without regularization), and RNN models.
    - Reporting final performance metrics.
- **Dependencies**: Listed in the imports section of the notebook.
- **Results**: Summarized in the notebook under "Summary of Results."

## Dependencies
The project relies on the following Python libraries:
- `numpy`: Numerical computations.
- `matplotlib`: Plotting and visualization.
- `librosa`: Audio processing and feature extraction.
- `sklearn`: Train-test split, confusion matrix, and metrics.
- `tensorflow.keras`: Building and training neural networks.
- `os`, `json`, `math`: File handling and utilities.

To install dependencies, run:
```bash
pip install numpy matplotlib librosa scikit-learn tensorflow
```

## Setup Instructions
1. **Download data**:
   ```python
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

    print("Path to dataset files:", path)
   ```

2. **Mount Google Drive**:
   - The notebook is designed to run in Google Colab.
   - Mount Google Drive to access the dataset:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Ensure the dataset is located at `/content/drive/MyDrive/Data/genres_original`.

3. **Run the Notebook**:
   - Open `Copy_of_ml_project_(modelling).ipynb` in Google Colab.
   - Execute cells sequentially to preprocess data, perform EDA, train models, and evaluate results.

## Data Preprocessing
- Audio files are loaded using `librosa.load()` with a sampling rate of 22,050 Hz.
- Waveform visualization is implemented using `librosa.display.waveshow()` to inspect audio characteristics.
- Features are extracted (not shown in the provided snippet but implied for model training).
- The dataset is split into training and testing sets using `sklearn.model_selection.train_test_split`.

## Models
Four models were implemented and evaluated:
1. **Deep Neural Network (DNN)**:
   - Accuracy: 0.570
   - Basic feedforward network, prone to overfitting.
2. **Convolutional Neural Network (CNN) without Regularization**:
   - Accuracy: 0.670
   - Improved performance over DNN but still overfit.
3. **Convolutional Neural Network (CNN) with Regularization**:
   - Accuracy: 0.772
   - Best-performing model, using techniques like dropout and L2 regularization to mitigate overfitting.
4. **Recurrent Neural Network (RNN) with Regularization**:
   - Accuracy: 0.609
   - Suitable for sequential data but less effective than CNN for this task.

**Note**: All models exhibited overfitting, a common challenge with neural networks. Regularization improved performance, particularly for the CNN.

## Results
The performance metrics for the models are summarized below:

| **Model**                        | **Accuracy** |
|----------------------------------|--------------|
| DNN                              | 0.570        |
| CNN (Without Regularization)     | 0.670        |
| CNN (With Regularization)        | 0.772        |
| RNN (With Regularization)        | 0.609        |

The CNN with regularization achieved the highest accuracy, balancing overfitting and generalization.

## Usage
To replicate the project:
1. Ensure the dataset is accessible in the specified Google Drive path.
2. Run the notebook in Google Colab, following the cell execution order.
3. Inspect the waveform visualizations and model performance metrics.
4. Modify hyperparameters or model architectures in the notebook to experiment further.

## Limitations
- **Overfitting**: All models overfit to some extent, indicating potential for improved regularization or data augmentation.
- **Dataset Size**: The GTZAN dataset is relatively small, which may limit model generalization.
- **Feature Extraction**: The provided snippet does not show feature extraction details (e.g., MFCCs, spectrograms), which are critical for model performance.

## Future Improvements
- Implement additional regularization techniques (e.g., data augmentation using `ImageDataGenerator` for spectrograms).
- Explore advanced architectures like transformers or hybrid CNN-RNN models.
- Increase dataset size or use pre-trained models for transfer learning.
- Add cross-validation to ensure robust performance metrics.

## Contact
For questions or feedback, please contact the author at aengusmartindonaire@gmail.com.
