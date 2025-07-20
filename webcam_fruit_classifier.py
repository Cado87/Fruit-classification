import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import argparse

class FruitClassifier:
    def __init__(self, model_path="models/fruit_recognition_model.keras"):
        """
        Initialize the fruit classifier with the trained model
        
        Args:
            model_path (str): Path to the saved Keras model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Mango', 'Grape', 'Plum', 'Kiwi', 'Pear', 'Apple', 'Orange', 'Banana',
            'Pomegranate', 'Strawberry', 'Pineapple', 'Fig', 'Peach', 'Apricot',
            'Avocado', 'Summer Squash', 'Lemon', 'Lime', 'Guava', 'Raspberry'
        ]
        self.input_size = (128, 128)  # Model was trained with 128x128 input size
        self.load_model()
        
    def load_model(self):
        """Load the trained Keras model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure the model file exists and is compatible.")
            return False
        return True
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize pixel values to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict(self, image, threshold=0.5):
        """
        Predict fruit classes from image (multi-label)
        
        Args:
            image: OpenCV image (BGR format)
            threshold (float): Probability threshold for classifying a fruit as present
        
        Returns:
            tuple: (list of predicted classes, list of (class, probability) tuples)
        """
        if self.model is None:
            return [], []
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]  # Already sigmoid, shape (20,)
        
        # Get all classes above threshold
        predicted_classes = [self.class_names[i] for i, prob in enumerate(probabilities) if prob > threshold]
        class_probs = [(self.class_names[i], float(prob)) for i, prob in enumerate(probabilities)]
        
        return predicted_classes, class_probs
    
    def classify_image(self, image_path, show_image=True, save_result=True):
        """
        Classify a single image file
        
        Args:
            image_path (str): Path to the image file
            show_image (bool): Whether to display the image with results
            save_result (bool): Whether to save the result image
        """
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return [], []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return [], []
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Make prediction
        predicted_classes, class_probs = self.predict(image)
        
        if predicted_classes:
            print(f"\n=== Classification Result ===")
            print(f"Predicted fruits: {', '.join(predicted_classes)}")
            print("Probabilities:")
            for cname, prob in class_probs:
                print(f"  {cname}: {prob:.2%}")
        else:
            print("No fruit detected above threshold.")
        
        # Show top 3 predictions
        top_3 = sorted(class_probs, key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 3 predictions:")
        for i, (cname, prob) in enumerate(top_3):
            print(f"{i+1}. {cname}: {prob:.2%}")
        
        if show_image:
            # Create display image with results
            display_image = image.copy()
            
            # Add text overlay for all predicted classes
            y0 = 50
            for idx, cname in enumerate(predicted_classes):
                prob = dict(class_probs)[cname]
                text = f"{cname}: {prob:.2%}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                text_x = 20
                text_y = y0 + idx * (text_size[1] + 10)
                cv2.rectangle(display_image, 
                              (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10),
                              (0, 0, 0), -1)
                color = (0, 255, 0) if prob > 0.8 else (0, 255, 255) if prob > 0.6 else (0, 165, 255)
                cv2.putText(display_image, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Add instructions
            cv2.putText(display_image, "Press any key to close", (20, display_image.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow('Fruit Classification Result', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save_result:
            # Save result image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{'_'.join(predicted_classes) if predicted_classes else 'None'}_{timestamp}.jpg"
            cv2.imwrite(result_filename, image)
            print(f"Result saved as: {result_filename}")
        
        return predicted_classes, class_probs

    def run_webcam(self, camera_index=0):
        """
        Run real-time fruit classification using webcam
        
        Args:
            camera_index (int): Camera device index (usually 0 for default camera)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera at index {camera_index}")
            return
        
        print("Webcam started! Press 'q' to quit, 's' to save image")
        print("Fruit classification running in real-time...")
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Create region of interest (ROI) for classification
            # Use center portion of the frame
            roi_size = min(width, height) // 2
            center_x, center_y = width // 2, height // 2
            roi_x1 = center_x - roi_size // 2
            roi_y1 = center_y - roi_size // 2
            roi_x2 = center_x + roi_size // 2
            roi_y2 = center_y + roi_size // 2
            
            # Draw ROI rectangle
            cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
            # Extract ROI for classification
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Make prediction
            predicted_classes, class_probs = self.predict(roi)
            
            # Display prediction results
            if predicted_classes:
                y0 = 30
                for idx, cname in enumerate(predicted_classes):
                    prob = dict(class_probs)[cname]
                    text = f"{cname}: {prob:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = 10
                    text_y = y0 + idx * (text_size[1] + 10)
                    cv2.rectangle(display_frame, 
                                  (text_x - 5, text_y - text_size[1] - 5),
                                  (text_x + text_size[0] + 5, text_y + 5),
                                  (0, 0, 0), -1)
                    color = (0, 255, 0) if prob > 0.8 else (0, 255, 255) if prob > 0.6 else (0, 165, 255)
                    cv2.putText(display_frame, text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                current_time = time.time()
                fps = 30 / (current_time - start_time)
                start_time = current_time
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Fruit Classifier', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"fruit_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam session ended.")

def main():
    """Main function to run the fruit classifier"""
    parser = argparse.ArgumentParser(description='Fruit Classification Application')
    parser.add_argument('--image', '-i', type=str, help='Path to image file for classification')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index for webcam mode (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Do not display image when classifying file')
    parser.add_argument('--no-save', action='store_true', help='Do not save result image when classifying file')
    
    args = parser.parse_args()
    
    print("=== Fruit Classification Application ===")
    print("This application uses a trained Keras model to classify fruits.")
    print()
    
    # Initialize classifier
    classifier = FruitClassifier()
    
    if classifier.model is None:
        print("Failed to load model. Please check the model file.")
        return
    
    print("Model loaded successfully!")
    print(f"Available classes: {', '.join(classifier.class_names)}")
    print()
    
    if args.image:
        # Image classification mode
        print(f"Classifying image: {args.image}")
        classifier.classify_image(
            args.image, 
            show_image=not args.no_display, 
            save_result=not args.no_save
        )
    else:
        # Webcam mode
        print("Camera options:")
        print("0 - Default camera")
        print("1 - External camera (if available)")
        print("2 - USB camera (if available)")
        
        try:
            camera_choice = input(f"Enter camera index (default: {args.camera}): ").strip()
            camera_index = int(camera_choice) if camera_choice else args.camera
        except ValueError:
            camera_index = args.camera
            print(f"Invalid input, using camera index {camera_index}")
        
        print(f"Starting webcam with camera index {camera_index}...")
        print("Press 'q' to quit, 's' to save current frame")
        print()
        
        # Run webcam classification
        classifier.run_webcam(camera_index)

if __name__ == "__main__":
    main() 