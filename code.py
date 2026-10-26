from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_FILE = os.path.join(BASE_DIR, "Training.csv")
TESTING_FILE = os.path.join(BASE_DIR, "Testing.csv")


# ============================================================
# SYMPTOMS
# ============================================================

l1 = [
    'back_pain',
    'constipation',
    'abdominal_pain',
    'diarrhoea',
    'mild_fever',
    'yellow_urine',
    'yellowing_of_eyes',
    'acute_liver_failure',
    'fluid_overload',
    'swelling_of_stomach',
    'swelled_lymph_nodes',
    'malaise',
    'blurred_and_distorted_vision',
    'phlegm',
    'throat_irritation',
    'redness_of_eyes',
    'sinus_pressure',
    'runny_nose',
    'congestion',
    'chest_pain',
    'weakness_in_limbs',
    'fast_heart_rate',
    'pain_during_bowel_movements',
    'pain_in_anal_region',
    'bloody_stool',
    'irritation_in_anus',
    'neck_pain',
    'dizziness',
    'cramps',
    'bruising',
    'obesity',
    'swollen_legs',
    'swollen_blood_vessels',
    'puffy_face_and_eyes',
    'enlarged_thyroid',
    'brittle_nails',
    'swollen_extremeties',
    'excessive_hunger',
    'extra_marital_contacts',
    'drying_and_tingling_lips',
    'slurred_speech',
    'knee_pain',
    'hip_joint_pain',
    'muscle_weakness',
    'stiff_neck',
    'swelling_joints',
    'movement_stiffness',
    'spinning_movements',
    'loss_of_balance',
    'unsteadiness',
    'weakness_of_one_body_side',
    'loss_of_smell',
    'bladder_discomfort',
    'foul_smell_of urine',
    'continuous_feel_of_urine',
    'passage_of_gases',
    'internal_itching',
    'toxic_look_(typhos)',
    'depression',
    'irritability',
    'muscle_pain',
    'altered_sensorium',
    'red_spots_over_body',
    'belly_pain',
    'abnormal_menstruation',
    'dischromic _patches',
    'watering_from_eyes',
    'increased_appetite',
    'polyuria',
    'family_history',
    'mucoid_sputum',
    'rusty_sputum',
    'lack_of_concentration',
    'visual_disturbances',
    'receiving_blood_transfusion',
    'receiving_unsterile_injections',
    'coma',
    'stomach_bleeding',
    'distention_of_abdomen',
    'history_of_alcohol_consumption',
    'blood_in_sputum',
    'prominent_veins_on_calf',
    'palpitations',
    'painful_walking',
    'pus_filled_pimples',
    'blackheads',
    'scurring',
    'skin_peeling',
    'silver_like_dusting',
    'small_dents_in_nails',
    'inflammatory_nails',
    'blister',
    'red_sore_around_nose',
    'yellow_crust_ooze'
]


# Remove duplicate symptoms while keeping the original order
l1 = list(dict.fromkeys(l1))


# ============================================================
# DISEASE NAMES
# ============================================================

disease = [
    'Fungal infection',
    'Allergy',
    'GERD',
    'Chronic cholestasis',
    'Drug Reaction',
    'Peptic ulcer diseae',
    'AIDS',
    'Diabetes',
    'Gastroenteritis',
    'Bronchial Asthma',
    'Hypertension',
    'Migraine',
    'Cervical spondylosis',
    'Paralysis (brain hemorrhage)',
    'Jaundice',
    'Malaria',
    'Chicken pox',
    'Dengue',
    'Typhoid',
    'hepatitis A',
    'Hepatitis B',
    'Hepatitis C',
    'Hepatitis D',
    'Hepatitis E',
    'Alcoholic hepatitis',
    'Tuberculosis',
    'Common Cold',
    'Pneumonia',
    'Dimorphic hemmorhoids(piles)',
    'Heart attack',
    'Varicose veins',
    'Hypothyroidism',
    'Hyperthyroidism',
    'Hypoglycemia',
    'Osteoarthristis',
    'Arthritis',
    '(vertigo) Paroymsal  Positional Vertigo',
    'Acne',
    'Urinary tract infection',
    'Psoriasis',
    'Impetigo'
]


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(TRAINING_FILE)
    tr = pd.read_csv(TESTING_FILE)

except FileNotFoundError:

    messagebox.showerror(
        "File Error",
        "Training.csv or Testing.csv was not found.\n\n"
        "Make sure both CSV files are in the same folder as clean_code.py."
    )

    raise SystemExit


# ============================================================
# CHECK SYMPTOM COLUMNS
# ============================================================

available_symptoms = []

for symptom in l1:

    if symptom in df.columns and symptom in tr.columns:
        available_symptoms.append(symptom)


if len(available_symptoms) == 0:

    messagebox.showerror(
        "Dataset Error",
        "No matching symptom columns were found in Training.csv."
    )

    raise SystemExit


# Use only symptoms that actually exist in both datasets
l1 = available_symptoms


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X = df[l1]

X_test = tr[l1]


# Convert disease names to numbers automatically
disease_names = sorted(df["prognosis"].astype(str).unique())

disease_to_number = {
    name: index
    for index, name in enumerate(disease_names)
}

number_to_disease = {
    index: name
    for name, index in disease_to_number.items()
}


y = df["prognosis"].astype(str).map(disease_to_number)
y_test = tr["prognosis"].astype(str).map(disease_to_number)


# ============================================================
# TRAIN MODELS
# ============================================================

print("Training models...")

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X, y)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

random_forest_model.fit(X, y)

naive_bayes_model = GaussianNB(var_smoothing=1e-5)
naive_bayes_model.fit(X, y)



# MODEL EVALUATION: CROSS-VALIDATION + HELD-OUT TEST SET
# ============================================================

models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Naive Bayes": naive_bayes_model
}

# Cross-validation compares models over multiple held-out folds of Training.csv.
# Testing.csv is kept separate for the final evaluation.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\\n5-FOLD CROSS-VALIDATION RESULTS (Training.csv)")
print("=" * 60)

for name, model in models.items():
    cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    cv_accuracy = accuracy_score(y, cv_pred)
    cv_precision = precision_score(
        y, cv_pred, average="weighted", zero_division=0
    )
    cv_recall = recall_score(
        y, cv_pred, average="weighted", zero_division=0
    )
    cv_f1 = f1_score(
        y, cv_pred, average="weighted", zero_division=0
    )

    print(f"\\n{name}")
    print("-" * 35)
    print(f"Accuracy  : {cv_accuracy * 100:.2f}%")
    print(f"Precision : {cv_precision * 100:.2f}%")
    print(f"Recall    : {cv_recall * 100:.2f}%")
    print(f"F1-Score  : {cv_f1 * 100:.2f}%")

print("\\nHELD-OUT TEST RESULTS (Testing.csv)")
print("=" * 60)

for name, model in models.items():
    # These models were fitted on all of Training.csv above.
    test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_recall = recall_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_pred, average="weighted", zero_division=0
    )

    print(f"\\n{name}")
    print("-" * 35)
    print(f"Accuracy  : {test_accuracy * 100:.2f}%")
    print(f"Precision : {test_precision * 100:.2f}%")
    print(f"Recall    : {test_recall * 100:.2f}%")
    print(f"F1-Score  : {test_f1 * 100:.2f}%")

print("\\n" + "=" * 60)


# ============================================================
# CREATE INPUT VECTOR
# ============================================================

def create_input_vector():

    selected_symptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get()
    ]

    # Remove None / empty values
    selected_symptoms = [
        symptom
        for symptom in selected_symptoms
        if symptom and symptom != "None"
    ]

    # Check if user selected anything
    if len(selected_symptoms) == 0:

        messagebox.showwarning(
            "No Symptoms",
            "Please select at least one symptom."
        )

        return None

    # Create completely new vector
    # This fixes the old l2 problem.
    input_data = [0] * len(l1)

    # Mark selected symptoms as 1
    for symptom in selected_symptoms:

        if symptom in l1:

            index = l1.index(symptom)
            input_data[index] = 1

    return [input_data]


# ============================================================
# DECISION TREE
# ============================================================

def DecisionTree():

    inputtest = create_input_vector()

    if inputtest is None:
        return

    prediction = decision_tree_model.predict(inputtest)[0]

    result = number_to_disease[int(prediction)]

    t1.delete("1.0", END)
    t1.insert(END, result)

    print("Decision Tree Prediction:", result)


# ============================================================
# RANDOM FOREST
# ============================================================

def randomforest():

    inputtest = create_input_vector()

    if inputtest is None:
        return

    prediction = random_forest_model.predict(inputtest)[0]

    result = number_to_disease[int(prediction)]

    t2.delete("1.0", END)
    t2.insert(END, result)

    print("Random Forest Prediction:", result)


# ============================================================
# NAIVE BAYES
# ============================================================

def NaiveBayes():

    inputtest = create_input_vector()

    if inputtest is None:
        return

    prediction = naive_bayes_model.predict(inputtest)[0]

    result = number_to_disease[int(prediction)]

    t3.delete("1.0", END)
    t3.insert(END, result)

    print("Naive Bayes Prediction:", result)


# ============================================================
# PREDICT ALL THREE
# ============================================================

def predict_all():

    inputtest = create_input_vector()

    if inputtest is None:
        return

    # Predictions
    dt_prediction = decision_tree_model.predict(inputtest)[0]
    rf_prediction = random_forest_model.predict(inputtest)[0]
    nb_prediction = naive_bayes_model.predict(inputtest)[0]

    # Convert numbers to disease names
    dt_result = number_to_disease[int(dt_prediction)]
    rf_result = number_to_disease[int(rf_prediction)]
    nb_result = number_to_disease[int(nb_prediction)]

    # Display Decision Tree
    t1.delete("1.0", END)
    t1.insert(END, dt_result)

    # Display Random Forest
    t2.delete("1.0", END)
    t2.insert(END, rf_result)

    # Display Naive Bayes
    t3.delete("1.0", END)
    t3.insert(END, nb_result)

    # Print results
    print()
    print("Prediction Results")
    print("-------------------------")
    print("Decision Tree :", dt_result)
    print("Random Forest :", rf_result)
    print("Naive Bayes   :", nb_result)
    print()


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_all():

    Symptom1.set("None")
    Symptom2.set("None")
    Symptom3.set("None")
    Symptom4.set("None")
    Symptom5.set("None")

    Name.set("")

    t1.delete("1.0", END)
    t2.delete("1.0", END)
    t3.delete("1.0", END)


# ============================================================
# GUI
# ============================================================

root = Tk()

root.title("Disease Prediction Using Machine Learning")

root.geometry("1200x750")

root.configure(background="blue")


# ============================================================
# VARIABLES
# ============================================================

Symptom1 = StringVar()
Symptom2 = StringVar()
Symptom3 = StringVar()
Symptom4 = StringVar()
Symptom5 = StringVar()

Name = StringVar()


Symptom1.set("None")
Symptom2.set("None")
Symptom3.set("None")
Symptom4.set("None")
Symptom5.set("None")


# ============================================================
# TITLE
# ============================================================

title = Label(
    root,
    text="Disease Predictor using Machine Learning",
    fg="white",
    bg="blue",
    font=("Arial", 30, "bold")
)

title.grid(
    row=0,
    column=0,
    columnspan=4,
    pady=20
)


subtitle = Label(
    root,
    text="Machine Learning Based Disease Prediction",
    fg="white",
    bg="blue",
    font=("Arial", 18)
)

subtitle.grid(
    row=1,
    column=0,
    columnspan=4,
    pady=5
)


# ============================================================
# PATIENT NAME
# ============================================================

NameLb = Label(
    root,
    text="Patient Name",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

NameLb.grid(
    row=3,
    column=0,
    padx=30,
    pady=20,
    sticky=W
)


NameEn = Entry(
    root,
    textvariable=Name,
    width=35,
    font=("Arial", 12)
)

NameEn.grid(
    row=3,
    column=1,
    pady=20
)


# ============================================================
# SYMPTOMS
# ============================================================

S1Lb = Label(
    root,
    text="Symptom 1",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

S1Lb.grid(row=4, column=0, padx=30, pady=10, sticky=W)


S2Lb = Label(
    root,
    text="Symptom 2",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

S2Lb.grid(row=5, column=0, padx=30, pady=10, sticky=W)


S3Lb = Label(
    root,
    text="Symptom 3",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

S3Lb.grid(row=6, column=0, padx=30, pady=10, sticky=W)


S4Lb = Label(
    root,
    text="Symptom 4",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

S4Lb.grid(row=7, column=0, padx=30, pady=10, sticky=W)


S5Lb = Label(
    root,
    text="Symptom 5",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

S5Lb.grid(row=8, column=0, padx=30, pady=10, sticky=W)


# ============================================================
# DROPDOWN OPTIONS
# ============================================================

OPTIONS = ["None"] + sorted(l1)


S1En = OptionMenu(
    root,
    Symptom1,
    *OPTIONS
)

S1En.config(width=30)

S1En.grid(row=4, column=1, pady=10)


S2En = OptionMenu(
    root,
    Symptom2,
    *OPTIONS
)

S2En.config(width=30)

S2En.grid(row=5, column=1, pady=10)


S3En = OptionMenu(
    root,
    Symptom3,
    *OPTIONS
)

S3En.config(width=30)

S3En.grid(row=6, column=1, pady=10)


S4En = OptionMenu(
    root,
    Symptom4,
    *OPTIONS
)

S4En.config(width=30)

S4En.grid(row=7, column=1, pady=10)


S5En = OptionMenu(
    root,
    Symptom5,
    *OPTIONS
)

S5En.config(width=30)

S5En.grid(row=8, column=1, pady=10)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = Button(
    root,
    text="PREDICT DISEASE",
    command=predict_all,
    bg="green",
    fg="white",
    font=("Arial", 14, "bold"),
    width=20,
    height=2
)

predict_button.grid(
    row=10,
    column=1,
    pady=25
)


# ============================================================
# CLEAR BUTTON
# ============================================================

clear_button = Button(
    root,
    text="CLEAR",
    command=clear_all,
    bg="red",
    fg="white",
    font=("Arial", 12, "bold"),
    width=15
)

clear_button.grid(
    row=10,
    column=2,
    pady=25
)


# ============================================================
# RESULTS
# ============================================================

result_title = Label(
    root,
    text="PREDICTION RESULTS",
    fg="white",
    bg="blue",
    font=("Arial", 18, "bold")
)

result_title.grid(
    row=12,
    column=0,
    columnspan=3,
    pady=15
)


# Decision Tree

dt_label = Label(
    root,
    text="Decision Tree",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

dt_label.grid(
    row=13,
    column=0,
    padx=30,
    pady=8,
    sticky=W
)


t1 = Text(
    root,
    height=1,
    width=35,
    bg="white",
    fg="black",
    font=("Arial", 12)
)

t1.grid(
    row=13,
    column=1,
    pady=8
)


# Random Forest

rf_label = Label(
    root,
    text="Random Forest",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

rf_label.grid(
    row=14,
    column=0,
    padx=30,
    pady=8,
    sticky=W
)


t2 = Text(
    root,
    height=1,
    width=35,
    bg="white",
    fg="black",
    font=("Arial", 12)
)

t2.grid(
    row=14,
    column=1,
    pady=8
)


# Naive Bayes

nb_label = Label(
    root,
    text="Naive Bayes",
    fg="yellow",
    bg="black",
    font=("Arial", 12, "bold")
)

nb_label.grid(
    row=15,
    column=0,
    padx=30,
    pady=8,
    sticky=W
)


t3 = Text(
    root,
    height=1,
    width=35,
    bg="white",
    fg="black",
    font=("Arial", 12)
)

t3.grid(
    row=15,
    column=1,
    pady=8
)


# ============================================================
# DISCLAIMER
# ============================================================

disclaimer = Label(
    root,
    text="This is an educational ML project and not a medical diagnosis.",
    fg="white",
    bg="blue",
    font=("Arial", 10)
)

disclaimer.grid(
    row=17,
    column=0,
    columnspan=3,
    pady=20
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()