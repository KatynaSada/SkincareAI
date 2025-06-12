# SkincareAI

Welcome to the SkinCareAI repository! This application is designed to provide high school students with a real-world example of how machine learning can be applied in the medical field. By analyzing facial images along with lifestyle details, the app offers personalized skincare recommendations.

## Overview

SkincareAI uses state-of-the-art models to identify:
- **Skin Type:** Whether your skin is dry, normal, or oily.
- **Acne Levels:** The presence and severity of acne.
- **Wrinkle Detection:** Analysis to determine wrinkle levels.

The application is built with [Streamlit](https://streamlit.io/), making it interactive and easy to use.

## How It Works

1. **Image Processing:**  
    The app accepts a face image uploaded by the user. It utilizes various image processing techniques:
    - Feature extraction using color histograms, local binary patterns, and Haralick features.
    - Deep learning features using the VGG16 architecture.

2. **Machine Learning Models:**  
    Different pre-trained models are applied to the extracted features:
    - **Skin Type Classification** using SVM.
    - **Acne Detection** using both SVM and deep learning features.
    - **Wrinkle Analysis** using specialized image processing.

3. **Lifestyle Integration:**  
    Users provide additional lifestyle details (age, profession, work hours, free time, and current product use) which are factored into the final recommendations.

4. **Personalized Recommendations:**  
    Based on the analysis, the app generates a list of suggestions tailored to your skin type and lifestyle.

## Getting Started

### Prerequisites
- Python 3.x
- Necessary libraries: Streamlit, OpenCV, NumPy, joblib, Mahotas, scikit-image, Matplotlib, TensorFlow, and more.

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/katynasada/SkinCarePredictor.git
    ```
2. Navigate to the project directory:
    ```
    cd SkinCarePredictor
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```
4. Run the app using Streamlit:
    ```
    streamlit run README.md
    ```

## Usage

1. **Upload Your Image:**  
    Use the file uploader to select a face image (JPG, JPEG, or PNG).

2. **Fill in Your Details:**  
    Enter your age, choose your profession, and input your work hours and free time. Also, indicate if you are already using skincare products.

3. **Generate Recommendations:**  
    Click the "Generate Recommendations" button to see your personalized skincare analysis and tips.

## Educational Value

This project offers a comprehensive example of:
- How machine learning can be applied to solve real-world problems.
- The integration of image processing and predictive analytics.
- The process of combining technical analysis with lifestyle and user data to provide meaningful recommendations.

Perfect as an educational tool, the SkincareAI demonstrates both the potential and the practicality of machine learning in a field like dermatology.

## Live Demo

Try the application online and experience the analysis firsthand!  
[Launch SkincareAI](https://skincareanalysis-vwymbk3mtyfgzbwzafcev6.streamlit.app/)

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

### Acknowledgments

This project was inspired by the approach from [Skin Care Analysis](https://github.com/harishvicky-23/Skin_care_analysis/tree/main).