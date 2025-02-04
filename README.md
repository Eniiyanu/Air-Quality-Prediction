# Air Quality Prediction from Low-Cost IoT Devices

Welcome to the Air Quality Prediction project! This repository contains the solution for predicting CO2 levels using low-cost IoT devices. The project is part of a hackathon hosted on Zindi, where I emerged third place.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Evaluation](#evaluation)
- [Submission](#submission)
- [Running on Azure ML Studio](#running-on-azure-ml-studio)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Carbon emissions significantly contribute to climate change, and monitoring these emissions is essential for mitigating environmental impact. High-accuracy reference meters are often prohibitively expensive. Chemotronix, a company developing low-cost sensors, seeks to build machine learning models that can accurately map sensor readings to CO2 levels measured by reference meters. This will enable affordable and scalable solutions for tracking carbon emissions globally.

The objective of this project is to develop a machine learning model that accurately predicts CO2 levels using data from Chemotronix’s low-cost sensors. By achieving this, I aim to bridge the gap between affordability and precision in carbon emission tracking, enabling widespread adoption of low-cost monitoring technologies.

### Goals

- Democratize access to environmental monitoring tools.
- Assist governments and organizations in implementing data-driven policies to curb carbon emissions.
- Promote sustainability by making emission tracking affordable for communities and industries worldwide.

## Dataset

The dataset is sourced from an experiment conducted by Chemotronix. The company deployed three prototype devices—Alpha, Beta, and Charlie—to collect sensor readings alongside a high-accuracy reference meter. These devices were tested in varying environmental conditions to ensure data diversity and robustness.

### Files

- `data/train.csv`: Training data with sensor readings and CO2 levels.
- `data/test.csv`: Test data with sensor readings only.
- `data/sample_submission.csv`: Sample submission file format.

## Project Structure

The repository is structured as follows:

```
.
├── data
│   ├── train.csv
│   ├── test.csv
|   ├── Variable Description.csv
│   └── sample_submission.csv
|
├── notebooks
│   └── air_quality_prediction.ipynb
├── src
│   ├── azure_train.py
│   └── train.py
|
├── outputs
│   └── submission.csv
├── documentation.md
├── requirements.txt
└── README.md
```

- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `src/`: Source code for data preprocessing, feature engineering, model training, Azure ML Studio integration, and utility functions.
- `outputs/`: Directory for storing submission files.
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project documentation.

## Feature Engineering

Feature engineering is a crucial step in this project. I create new features from the sensor readings to improve the model's performance. The following features are engineered:

- Basic sensor statistics (mean, standard deviation, median, max, min, range).
- Advanced statistical features (skewness, kurtosis).
- Ratio features between different sensors.
- Temperature and humidity compensation features.
- Environmental interaction features.

## Model Training

I use the RandomForestRegressor algorithm for training the predictive model. Random Forest is chosen for its efficiency and performance with structured data. The training process includes:

1. Loading and preprocessing the data.
2. Creating advanced features.
3. Imputing missing values.
4. Scaling the features.
5. Training the model using K-Fold cross-validation.


## Hyperparameter Optimization

I use Optuna for hyperparameter optimization. Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It provides a way to optimize the model hyperparameters efficiently.

## Evaluation

The evaluation metric for this competition is Root Mean Squared Error (RMSE). RMSE is a standard way to measure the error of a model in predicting quantitative data.

## Submission

The final predictions are saved in a CSV file (submission.csv) in the `outputs/` directory. The submission file follows the format provided in `sample_submission.csv`.

### Submission Format

```
ID              Target
ID_C7AV4GEJP9    605.5
ID_AFVZYGLXXY    598
```

## Running on Azure ML Studio

This project can be run on Microsoft Azure ML Studio using the provided scripts:

- `azure_train.py`: Script to train the model on Azure ML Studio.
- `train.py`: Script to train the model locally.

### Steps to Run on Azure ML Studio

1. Set up an Azure ML Studio workspace.
2. Upload the dataset to the workspace.
3. Create a compute instance for running the training script.
4. Upload `azure_train.py` and the required dependencies to the workspace.
5. Run the `azure_train.py` script on the compute instance to train the model.
6. The trained model and results will be saved in the specified Azure storage.


## Requirements

The project requires the following dependencies:

- pandas==1.5.0
- numpy==1.23.0
- scikit-learn==1.1.2
- xgboost==1.6.1
- optuna==3.0.0
- azureml-sdk==1.34.0

To install the dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To run the project:

1. Clone the repository:
    ```sh
    git clone https://github.com/Eniiyanu/Zindi-Air-Prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Zindi-Air-Prediction
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebooks for exploratory data analysis and model training:
    ```sh
    jupyter notebook notebooks/air_quality_prediction.ipynb
    ```
5. Generate the final submission file by running the model training script locally:
    ```sh
    python src/train.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions, please contact me.

