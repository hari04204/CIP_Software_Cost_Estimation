# DASCEM Model with Hybrid ACO-PSO Optimization & Baseline Comparisons

## Overview

This project implements a cost estimation framework that compares three different models:

1. **COCOMO-style Model:**  
   A log–log regression model that predicts project cost based on *Size_FunctionPoints*. It uses polynomial expansion and Ridge regression, optimized via GridSearchCV.

2. **Agile Story Points Model:**  
   A linear regression model that predicts project cost based on *Size_StoryPoints*, optimized with polynomial expansion and Ridge regression using GridSearchCV.

3. **DASCEM Model:**  
   An Adaptive Neuro-Fuzzy Inference System (ANFIS)-based model that uses two features (*Size_FunctionPoints* and *Team_Size*) to predict project cost. Its parameters are optimized using a hybrid Particle Swarm Optimization (PSO) algorithm enhanced with an Ant Colony Optimization (ACO)-inspired reinitialization strategy. GPU acceleration via PyTorch is used to speed up the optimization process.

## Problems with Baseline Models

Traditional models (e.g., COCOMO-style and Agile Story Points models) suffer from several limitations:

- **Limited Flexibility:**  
  They assume fixed functional forms (e.g., linear or log–log) which may not capture the true, often non-linear, relationships in modern software projects.

- **Lack of Adaptability:**  
  They do not adapt dynamically to changes in project conditions or incorporate multiple interacting factors without extensive manual calibration.

- **Handling Uncertainty:**  
  Traditional approaches often ignore uncertainty in project inputs and environmental factors, resulting in higher estimation errors.

- **Static Parameter Calibration:**  
  Baseline models typically calibrate parameters using historical data and then fix them, limiting responsiveness to evolving project contexts.

## Motivation and Derivation of the DASCEM Model

To address these issues, the DASCEM model was derived using an ANFIS framework. The model blends fuzzy logic with adaptive linear regression to capture non-linear relationships, incorporate uncertainty, and dynamically adapt to changing project conditions.

### Mathematical Formulation

#### 1. Fuzzification

For each input \( x_i \) (e.g., *Size_FunctionPoints* or *Team_Size*), we define \( M \) Gaussian membership functions:

$$
\mu_{i,j}(x_i) = \exp\left(-\frac{(x_i - c_{i,j})^2}{2\sigma_{i,j}^2 + \epsilon}\right)
$$

- \( c_{i,j} \): Center of the \( j \)-th membership function for the \( i \)-th input.
- \( \sigma_{i,j} \): Spread (standard deviation) of the \( j \)-th membership function.
- \( \epsilon \): A small constant to prevent division by zero.

#### 2. Rule Formation

Each fuzzy rule \( r \) is defined by a combination of membership functions for each input:

$$
r = (j_1, j_2, \dots, j_n)
$$

- \( n \): Number of inputs.
- Total number of rules: \( M^n \).

#### 3. Firing Strength Calculation

The firing strength \( f_r \) of rule \( r \) is computed as the product of the membership degrees for each input:

$$
f_r = \prod_{i=1}^{n} \mu_{i,j_i}(x_i)
$$

The normalized firing strength is:

$$
\bar{f}_r = \frac{f_r}{\sum_{r'} f_{r'}}
$$

#### 4. Consequent Output

Each rule \( r \) has a linear consequent function:

$$
y_r = \sum_{i=1}^{n} a_{r,i}\, x_i + b_r
$$

- \( a_{r,i} \): Coefficient for input \( x_i \) in rule \( r \).
- \( b_r \): Bias term for rule \( r \).

#### 5. Final Output

The overall predicted log–scale cost is the weighted sum of rule outputs:

$$
\hat{y} = \sum_{r=1}^{M^n} \bar{f}_r \, y_r
$$

To convert the log–scale prediction back to the original cost scale:

$$
\text{Cost} = \exp(\hat{y})
$$

#### 6. Optimization Objective

The model parameters \( \{c_{i,j}, \sigma_{i,j}, a_{r,i}, b_r\} \) are optimized to minimize the RMSE on the log–transformed target:

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \log(\text{Cost}_i) - \hat{y}_i \right)^2}
$$

where \( N \) is the number of projects.

## Implementation Details

- **Data Preprocessing:**  
  Inputs (*Size_FunctionPoints* and *Team_Size*) are normalized using min–max scaling. The target (Actual_Cost_INR) is log–transformed.

- **Model Architecture:**  
  The `DASCEMModel` class implements the ANFIS-based model, using Gaussian membership functions, rule formation, and weighted linear consequents.

- **Optimization:**  
  A hybrid PSO algorithm with ACO-inspired reinitialization is used to optimize the model parameters. The optimizer minimizes RMSE on the log–transformed target. GPU acceleration via PyTorch is used where available.

- **Baseline Models:**  
  The COCOMO-style and Agile Story Points models are built using scikit‑learn pipelines, with hyperparameter tuning (via GridSearchCV) over polynomial degree and Ridge regression regularization strength.

## Usage

1. **Training the Models:**  
   - Run `train_cocomo.py` to train and save the COCOMO model (e.g., as `cocomo_model.pkl`).
   - Run `train_agile.py` to train and save the Agile Story Points model (e.g., as `agile_model.pkl`).
   - Run `train_dascem.py` to optimize the DASCEM model parameters using hybrid ACO-PSO and save them (e.g., as `dascem_model_optimized.pt`).

2. **Model Comparison:**  
   Run `compare_models.py` to load all three models, generate predictions, compute RMSE values (in original cost units), and visualize Actual vs. Predicted costs for each model.

## Conclusion

The DASCEM model was developed to overcome the limitations of traditional cost estimation models by:

- Capturing non-linear relationships through fuzzy membership functions.
- Incorporating uncertainty and interacting factors via rule-based fuzzy inference.
- Dynamically adapting model parameters using a hybrid PSO algorithm with ACO-inspired reinitialization.

By comparing the RMSE and visual predictions of the DASCEM model against baseline models (COCOMO-style and Agile Story Points), the project demonstrates the potential of this neuro-fuzzy approach for improved software cost estimation.

---

This document includes detailed mathematical formulas and explanations. When you paste this into an MD editor that supports LaTeX (for example, GitHub or a Jupyter Notebook), the formulas will render properly. Feel free to expand on each section as needed for your audience.
