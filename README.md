# Yield Prediction System

## Overview
The Yield Prediction System is a comprehensive project that predicts agricultural yield using a trained neural network model. This system integrates advanced machine learning techniques, data preprocessing, model evaluation, visualization, and a user-friendly GUI for easy interaction.

## Features
- **Data Preprocessing**:
  - Label Encoding for categorical features.
  - Standardization for numerical features.
  - Splitting dataset into training, validation, and testing sets.
- **Dynamic Dropout Evaluation**: Tests multiple dropout rates to prevent overfitting and optimize performance.
- **Neural Network**:
  - Feedforward architecture with two hidden layers.
  - Adjustable dropout layers.
- **Performance Metrics**: Mean Squared Error (MSE) and Mean Absolute Error (MAE) for evaluation.
- **Visualization**:
  - True vs Predicted scatter plot.
  - Residuals plot.
  - Training and validation loss curve.
  - Feature importance using permutation importance.
- **Graphical User Interface (GUI)**: Developed using Tkinter, allowing users to input features and receive predictions.

## Dataset
The dataset includes agricultural and environmental data with the following key columns:
- `Area`: Represents the geographical region (categorical).
- `Item`: Represents the crop or agricultural product (categorical).
- `average_rain_fall_mm_per_year`: Average rainfall in millimeters per year (numerical).
- `pesticides_tonnes`: Usage of pesticides in tonnes (numerical).
- `avg_temp`: Average temperature (numerical).
- `Year`: Year of observation (numerical).

### Preprocessing Steps
1. **Label Encoding**: Converts categorical columns (`Area` and `Item`) into numerical representations using `LabelEncoder`.
2. **Feature Scaling**: Standardizes numerical columns to ensure uniform scaling.
3. **Data Splitting**: Divides the dataset into training, validation, and test sets with an 80-10-10 ratio.

## Neural Network Model
- **Architecture**:
  - Input Layer: Takes standardized features.
  - Hidden Layers: Two layers with 128 and 64 neurons, ReLU activation.
  - Dropout Layers: Dynamically adjusted rates to mitigate overfitting.
  - Output Layer: Single neuron for yield prediction.
- **Hyperparameters**:
  - Optimizer: Adam with a learning rate of 0.001.
  - Loss Function: Mean Squared Error (MSE).
  - Metrics: Mean Absolute Error (MAE).
  - Epochs: 50.
  - Batch Size: 32.

### Results
- Best Dropout Rate: `0.1`
- Performance Metrics:
  - Mean Squared Error (MSE): 0.1972
  - Mean Absolute Error (MAE): 0.2957

## Visualizations
1. **True vs Predicted Values**: Scatter plot comparing actual and predicted yields.
2. **Residuals Plot**: Residuals against predicted values to evaluate prediction quality.
3. **Learning Curve**: Training and validation loss over epochs.
4. **Feature Importance**: Permutation importance analysis to highlight impactful features.

## Graphical User Interface (GUI)
A Tkinter-based GUI application for user-friendly predictions:
- **Input Fields**: Users can input values for features like `Area`, `Item`, `average_rain_fall_mm_per_year`, etc.
- **Output**: Displays raw input data and predicted yield.
- **Validation**: Ensures valid inputs for categorical and numerical features.

### Usage Instructions
1. Run the script to launch the GUI.
2. Enter feature values in the input fields.
3. Click "Predict Yield" to see the prediction result.

## How to Run the Project
1. Clone the repository and install dependencies:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
   ```
2. Prepare your dataset with the structure described above and save it as `yield_df.csv`.
3. Train the model by running the training script.
4. Use the Tkinter GUI script to make predictions.

## Future Improvements
- Add support for additional features like soil quality, humidity, and other environmental factors.
- Implement advanced models like CNNs or LSTMs.
- Perform hyperparameter optimization using Grid Search or Bayesian methods.

## Contact
For questions, feedback, or collaborations, please reach out at salmaayman.mokhtar@gmail.com.

