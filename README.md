# Yield Prediction App

This project is a web-based application for predicting agricultural yields using machine learning. It allows users to input specific features about a region and crop, such as average rainfall, pesticide usage, and temperature, to estimate the yield.

## Features

- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive user experience.
- **Accurate Predictions**: Leverages a pre-trained neural network model to predict crop yield.
- **Dynamic Inputs**: Allows users to select regions and crops from preloaded datasets.
- **Scalable Architecture**: Utilizes scalable machine learning pipelines for preprocessing and predictions.

## How It Works

1. **Input Fields**:
   - **Area**: Select the region from a predefined list.
   - **Item**: Choose the crop or item to predict yield for.
   - **Average Rainfall (mm/year)**: Specify the average rainfall.
   - **Pesticides (tonnes)**: Enter the amount of pesticides used.
   - **Average Temperature (°C)**: Provide the average temperature for the area.
   - **Year**: Specify the year for prediction.

2. **Preprocessing**:
   - Encodes categorical inputs using `LabelEncoder`.
   - Scales numerical features using `StandardScaler`.

3. **Prediction**:
   - Combines encoded and scaled features into a single input array.
   - Feeds the array into a pre-trained neural network model (`final_model.h5`).
   - The output is scaled back to human-readable form using a saved scaler.

4. **Results**:
   - Displays the predicted yield in hectograms per hectare (hg/ha).

## Prerequisites

- Python 3.7+
- Libraries:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
  - `pickle`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/yield-prediction-app.git
   cd yield-prediction-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the required model and preprocessing files in the root directory:
   - `final_model.h5`
   - `label_encoder_area.pkl`
   - `label_encoder_item.pkl`
   - `scaler_features.pkl`
   - `scaler_target.pkl`
   - `yield_df.csv`

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in the terminal to access the app.

3. Enter the required details and click **Predict Yield** to get the results.

## Dataset

The application uses a dataset (`yield_df.csv`) containing information about regions, crops, and related features.

## File Structure

- `app.py`: Main Streamlit application file.
- `final_model.h5`: Pre-trained model for yield prediction.
- `*.pkl`: Preprocessing files for encoding and scaling.
- `yield_df.csv`: Dataset used for extracting area and crop options.

## Future Improvements

- Support for additional crops and regions.
- Integration of real-time weather data.
- Enhanced visualization for input data and predictions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- TensorFlow for model development.
- Streamlit for creating the interactive web application.
- Contributors to open-source libraries used in the project.
