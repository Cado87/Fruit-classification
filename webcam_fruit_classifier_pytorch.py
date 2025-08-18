import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os
import argparse
import json

class FruitClassifierPyTorch:
    def __init__(self, model_path="models/fruit_classifier_model.pth"):
        """
        Initialize the fruit classifier with the trained PyTorch model
        
        Args:
            model_path (str): Path to the saved PyTorch model
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (224, 224)  # Model was trained with 224x224 input size
        
        # Define the transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Try to load class names from file if available; otherwise infer later from the model
        self.class_names = self._load_class_names_from_file() or []
        
        self.load_model()
        
    def load_model(self):
        """Load the trained PyTorch model"""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Load checkpoint/state dict first to infer expected number of classes
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = None
            if isinstance(checkpoint, dict):
                # Common nesting patterns
                for candidate_key in ["state_dict", "model_state_dict", "model"]:
                    if candidate_key in checkpoint and isinstance(checkpoint[candidate_key], dict):
                        state_dict = checkpoint[candidate_key]
                        break
                if state_dict is None:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Try to detect the expected out_features from fc.weight in the state dict
            expected_num_classes = None
            fc_weight_key = None
            for key in state_dict.keys():
                if key.endswith("fc.weight") or key.endswith("module.fc.weight"):
                    fc_weight_key = key
            if fc_weight_key is not None:
                expected_num_classes = state_dict[fc_weight_key].shape[0]

            # Create the model architecture (ResNet-18)
            self.model = models.resnet18(pretrained=False)

            # Determine final layer size
            if expected_num_classes is None:
                # Fallback to provided class names length or default to 165 if unknown
                expected_num_classes = len(self.class_names) if self.class_names else 165

            # Ensure class_names length matches expected_num_classes
            if not self.class_names:
                # No names available; generate placeholders
                self.class_names = [f"Class_{i}" for i in range(expected_num_classes)]
            elif len(self.class_names) != expected_num_classes:
                # Reconcile mismatch by truncating or padding with placeholders
                original_len = len(self.class_names)
                if original_len > expected_num_classes:
                    self.class_names = self.class_names[:expected_num_classes]
                else:
                    self.class_names.extend([f"Class_{i}" for i in range(original_len, expected_num_classes)])
                print(f"Warning: Adjusted class_names length from {original_len} to {expected_num_classes} to match model outputs.")

            # Replace the final layer to match the expected number of classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, expected_num_classes)

            # Load the saved state dictionary (allow non-critical mismatches)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"Warning: load_state_dict with non-strict mode. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            
            # Set model to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            
            print("Model loaded successfully!")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure the model file exists and is compatible.")
            return False
        return True
    
    def _load_class_names_from_file(self):
        """Attempt to load class names from a JSON or TXT file in the models directory."""
        try:
            base_dir = os.path.dirname(self.model_path)
            # JSON format: ["apple", "banana", ...]
            json_path = os.path.join(base_dir, "class_names.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    names = json.load(f)
                if isinstance(names, list) and all(isinstance(n, str) for n in names):
                    return names
            # TXT format: one class name per line
            txt_path = os.path.join(base_dir, "class_names.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    return names
        except Exception as e:
            print(f"Warning: Failed to load class names from file: {e}")
        return None
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image tensor ready for model prediction
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply the transform
        image_tensor = self.transform(pil_image)
        
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0)
        
        return image_batch
    
    def predict(self, image):
        """
        Predict fruit class from image
        
        Args:
            image: OpenCV image (BGR format)
        
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        if self.model is None:
            return None, 0.0
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Move to device
        processed_image = processed_image.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
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
            return None, 0.0
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, 0.0
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Make prediction
        predicted_class, confidence = self.predict(image)
        
        if predicted_class:
            print(f"\n=== Classification Result ===")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print("No prediction made.")
        
        if show_image:
            # Create display image with results
            display_image = image.copy()
            
            # Add text overlay
            text = f"{predicted_class}: {confidence:.2%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = 20
            text_y = 50
            
            # Draw background rectangle
            cv2.rectangle(display_image, 
                          (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          (0, 0, 0), -1)
            
            # Choose color based on confidence
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
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
            result_filename = f"result_{predicted_class}_{timestamp}.jpg"
            cv2.imwrite(result_filename, image)
            print(f"Result saved as: {result_filename}")
        
        return predicted_class, confidence

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
            predicted_class, confidence = self.predict(roi)
            
            # Display prediction results
            if predicted_class:
                text = f"{predicted_class}: {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = 10
                text_y = 30
                
                # Draw background rectangle
                cv2.rectangle(display_frame, 
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)
                
                # Choose color based on confidence
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
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
            cv2.imshow('Fruit Classifier (PyTorch)', display_frame)
            
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
    parser = argparse.ArgumentParser(description='Fruit Classification Application (PyTorch)')
    parser.add_argument('--image', '-i', type=str, help='Path to image file for classification')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index for webcam mode (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Do not display image when classifying file')
    parser.add_argument('--no-save', action='store_true', help='Do not save result image when classifying file')
    
    args = parser.parse_args()
    
    print("=== Fruit Classification Application (PyTorch) ===")
    print("This application uses a trained PyTorch model to classify fruits.")
    print()
    
    # Initialize classifier
    classifier = FruitClassifierPyTorch()
    
    if classifier.model is None:
        print("Failed to load model. Please check the model file.")
        return
    
    print("Model loaded successfully!")
    print(f"Number of classes: {len(classifier.class_names)}")
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