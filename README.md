# Fruit-classification
Tests with several fruit classificators


DeepFruit
Web: https://pmc.ncbi.nlm.nih.gov/articles/PMC10507127/#refdata001
Dataset: https://data.mendeley.com/datasets/5prc54r4rt/1


# Conda environment

First, let's create a new Conda environment. Open your terminal and run the following command:
conda create --name fruit-classification python=3.11 -y

To see environments created:
conda env list

Activate the new environment:
conda activate fruit-classification

Install dependencies (first time):
pip install tensorflow opencv-python numpy pillow

Run application:
python webcam_fruit_classifier.py
 


# Fruit Classification Webcam Application

This application allows you to classify fruits in real-time using your computer's webcam and a pre-trained Keras model.

## Features

- **Real-time fruit classification** using webcam
- **20 fruit classes**: 'Mango', 'Grape', 'Plum', 'Kiwi', 'Pear', 'Apple', 'Orange', 'Banana', 'Pomegranate', 'Strawberry', 'Pineapple', 'Fig', 'Peach', 'Apricot', 'Avocado', 'Summer Squash', 'Lemon', 'Lime', 'Guava', 'Raspberry'
- **Confidence scoring** with color-coded results
- **FPS display** for performance monitoring
- **Image capture** functionality to save frames
- **Multiple camera support** for different camera devices

## Requirements

- Python 3.7 or higher
- Webcam or camera device
- The following Python packages (see `requirements.txt`):
  - tensorflow >= 2.10.0
  - opencv-python >= 4.5.0
  - numpy >= 1.21.0
  - Pillow >= 8.0.0

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify the model file exists:**
   Make sure the file `models/fruit_recognition_model.keras` is present in your project directory.

## Usage

### Quick Start

1. **Test the model first:**
   ```bash
   python test_model.py
   ```
   This will verify that the model can be loaded and make predictions.

2. **Run the webcam classifier:**
   ```bash
   python webcam_fruit_classifier.py
   ```

3. **Classify a single image:**
   ```bash
   python webcam_fruit_classifier.py --image path/to/your/image.jpg
   ```
   Or use the simplified wrapper:
   ```bash
   python classify_image.py path/to/your/image.jpg
   ```

### How to Use

#### Webcam Mode
1. **Camera Selection:**
   - The application will prompt you to select a camera index
   - Usually `0` is the default camera
   - Try `1` or `2` for external/USB cameras

2. **Using the Application:**
   - A green rectangle shows the region of interest (ROI) for classification
   - Hold fruits in the center of the frame for best results
   - The predicted fruit class and confidence score will appear at the top
   - Color coding:
     - 🟢 Green: High confidence (>80%)
     - 🟡 Yellow: Medium confidence (60-80%)
     - 🟠 Orange: Low confidence (50-60%)

3. **Controls:**
   - Press `q` to quit the application
   - Press `s` to save the current frame as an image

#### Image Classification Mode
1. **Command Line Options:**
   ```bash
   # Basic image classification
   python webcam_fruit_classifier.py --image fruit.jpg
   
   # Without displaying the image
   python webcam_fruit_classifier.py --image fruit.jpg --no-display
   
   # Without saving the result
   python webcam_fruit_classifier.py --image fruit.jpg --no-save
   
   # Using the simplified wrapper
   python classify_image.py fruit.jpg
   ```

2. **Features:**
   - Shows top 3 predictions with confidence scores
   - Displays the image with classification result overlay
   - Saves the result image with timestamp
   - Supports common image formats (JPG, PNG, etc.)

### Tips for Best Results

1. **Lighting:** Ensure good, even lighting on the fruit
2. **Positioning:** Place the fruit in the center green rectangle
3. **Distance:** Keep the fruit at a reasonable distance (not too close or far)
4. **Background:** Use a simple, uncluttered background
5. **Stability:** Hold the camera steady for more accurate predictions

## Troubleshooting

### Common Issues

1. **"Could not open camera"**
   - Try different camera indices (0, 1, 2)
   - Check if your webcam is working in other applications
   - On Windows, try running as administrator

2. **"Error loading model"**
   - Verify the model file exists at `models/fruit_recognition_model.keras`
   - Check if TensorFlow is properly installed
   - Run `python test_model.py` to diagnose issues

3. **Low FPS or lag**
   - Close other applications using the camera
   - Reduce the ROI size in the code if needed
   - Check your computer's performance

4. **Poor classification accuracy**
   - Ensure good lighting conditions
   - Position the fruit clearly in the center
   - Try different angles and distances
   - Make sure the fruit is one of the 20 supported classes

### Performance Optimization

- **GPU Support:** If you have a compatible GPU, TensorFlow will automatically use it
- **Model Optimization:** The model is optimized for real-time inference
- **Frame Skipping:** The application processes every frame, but you can modify it to skip frames for better performance

## File Structure

```
fruit-classification/
├── models/
│   └── fruit_recognition_model.keras    # Trained model
├── webcam_fruit_classifier.py           # Main application (webcam + image)
├── classify_image.py                    # Simple image classification wrapper
├── test_model.py                        # Model testing script
└── README.md                           # This file
```

## Technical Details

### Model Architecture
- Input size: 128x128 pixels
- Output: 20 classes (fruits)
- Preprocessing: RGB conversion, resizing, normalization

### Image Processing Pipeline
1. Capture frame from webcam
2. Extract Region of Interest (ROI)
3. Convert BGR to RGB
4. Resize to 128x128
5. Normalize pixel values to [0,1]
6. Make prediction
7. Display results with confidence score

### Performance Metrics
- FPS (Frames Per Second) display
- Confidence threshold filtering (>50%)
- Real-time processing with minimal latency

## Contributing

Feel free to modify the code for your specific needs:
- Adjust the ROI size for different camera setups
- Modify confidence thresholds
- Add new fruit classes (requires retraining the model)
- Implement additional features like recording or batch processing

## License

This project is part of the fruit classification research. Please refer to the original project documentation for licensing information.