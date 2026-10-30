# Disease Prediction Using Machine Learning

A Python-based disease prediction application that uses patient-selected
symptoms to generate predictions with three machine learning
classifiers: Decision Tree, Random Forest, and Gaussian Naive Bayes. The
project includes a desktop graphical user interface built with Tkinter.

> **Disclaimer:** This is an educational prototype. Its predictions are
> not medical diagnoses and must not replace advice, diagnosis, or
> treatment from a qualified healthcare professional.

## Project Overview

The application allows a user to: - Enter a patient name. - Select up to
five symptoms from dropdown menus. - Generate predictions using three
trained machine learning classifiers. - View each model's predicted
disease. - Clear the form and enter another set of symptoms.

## Machine Learning Models

1.  **Decision Tree Classifier** -- learns decision rules from symptom
    features to classify a disease category.
2.  **Random Forest Classifier** -- combines predictions from multiple
    decision trees.
3.  **Gaussian Naive Bayes** -- uses Bayes' theorem with a Gaussian
    feature-distribution assumption.

## Reported Model Performance

The project report records these evaluation results:

  Model                    Accuracy   Precision   Recall   F1-score
  ---------------------- ---------- ----------- -------- ----------
  Decision Tree              95.12%      92.68%   95.12%     93.50%
  Random Forest              90.24%      87.80%   90.24%     88.62%
  Gaussian Naive Bayes       95.12%      92.68%   95.12%     93.50%

These values are taken from the project report. Check the final source
code for the exact evaluation procedure, multiclass averaging method,
and data split or cross-validation configuration before describing those
details.

## Technology Stack

-   **Language:** Python
-   **GUI:** Tkinter
-   **Data handling:** Pandas, NumPy
-   **Machine learning:** Scikit-learn
-   **Dataset format:** CSV

The project imports Scikit-learn classifiers and evaluation utilities
including accuracy, precision, recall, F1-score, `StratifiedKFold`, and
`cross_val_predict`.

## Dataset

The project uses two CSV files:

-   `Training.csv` -- used for model training.
-   `Testing.csv` -- used for testing/evaluation.

The project report identifies symptom columns as input features and
`prognosis` as the target disease label.

The actual dataset source, number of records, number of symptom
features, and disease-class count are not specified here. Refer to the
actual CSV files for those details. Keep both CSV files in the location
expected by the Python program.

## Application Workflow

1.  Load the training and testing CSV files.
2.  Separate symptom columns (input features) from the `prognosis`
    target.
3.  Prepare the data in the format expected by the classifiers.
4.  Train the Decision Tree, Random Forest, and Gaussian Naive Bayes
    models.
5.  Evaluate the models using the configured evaluation procedure.
6.  Open the Tkinter application.
7.  Enter the patient name and select symptoms.
8.  Convert the selected symptoms into the feature representation
    expected by the trained models.
9.  Display the prediction from each model.

**Important:** GUI symptom selections must be mapped to the same feature
names and order used during model training.

## Application Interface

The Tkinter interface contains: - Patient name input field. - Five
symptom dropdown menus. - **Predict Disease** button. - **Clear**
button. - Separate result fields for Decision Tree, Random Forest, and
Naive Bayes.

## Requirements

-   Python 3.x
-   NumPy
-   Pandas
-   Scikit-learn
-   Tkinter (usually included with standard Python installations)

## Installation and Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and
folder name.

### 2. Create a virtual environment (recommended)

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

``` cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn
```

Tkinter is normally included with Python on Windows. If it is
unavailable, install the appropriate Tk support for your Python
distribution.

### 4. Check the project files

Ensure that the Python application file and required CSV files are
present, including:

``` text
Training.csv
Testing.csv
```

Place the CSV files at the paths expected by the application.

### 5. Run the application

Run the Python file containing your Tkinter application. For example, if
the filename is `main.py`:

``` bash
python main.py
```

Replace `main.py` with the actual filename in your repository.

## Evaluation Metrics

-   **Accuracy:** proportion of all predictions that are correct.
-   **Precision:** proportion of predicted positive cases that are
    correct for a class.
-   **Recall:** proportion of actual positive cases correctly identified
    for a class.
-   **F1-score:** harmonic mean of precision and recall.

For multiclass classification, the averaging method used for precision,
recall, and F1-score affects the reported values. Refer to the
evaluation code for the exact configuration.

## Team Members

  Member          Role
  --------------- ----------------------------------------
  B. Hemanth      Project Lead and Requirement Analysis
  I. Laxman Sai   Dataset and Data Preprocessing
  K. Manesh       Machine Learning Model Development
  V. Sri Vivek    GUI Development and System Integration

## Limitations

-   Predictions depend on the quality, coverage, and suitability of the
    dataset.
-   The model can predict only disease categories represented in its
    training data.
-   Different algorithms may produce different predictions for the same
    symptom combination.
-   This academic prototype has not been established as a clinically
    validated diagnostic system.

## Future Scope

Possible improvements, subject to further development and validation,
include: - Expanding and validating the dataset using reliable
sources. - Improving input validation and the user interface. - Testing
additional classification algorithms. - Improving model evaluation and
documenting the validation methodology. - Conducting appropriate
clinical validation before any real-world healthcare use.

## Acknowledgements

Developed as a B.Tech Computer Science and Engineering project at Aditya
University.
