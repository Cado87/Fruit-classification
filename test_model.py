import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def test_model_loading():
    """Test if the model can be loaded successfully"""
    model_path = "models/fruit_recognition_model.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        print(f"Testing model loading from {model_path}...")
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Test model summary
        print("\nModel Summary:")
        model.summary()
        
        # Test with dummy input
        dummy_input = np.random.random((1, 128, 128, 3))
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"\n✓ Model prediction shape: {prediction.shape}")
        print(f"✓ Number of classes: {prediction.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_preprocessing():
    """Test image preprocessing"""
    try:
        import cv2
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (128, 128))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        print("✓ Image preprocessing test passed!")
        print(f"  Input shape: {dummy_image.shape}")
        print(f"  Output shape: {image_batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocessing test: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Model Testing ===")
    print()
    
    # Test model loading
    model_ok = test_model_loading()
    print()
    
    # Test preprocessing
    preprocessing_ok = test_preprocessing()
    print()
    
    if model_ok and preprocessing_ok:
        print("✓ All tests passed! The model is ready to use.")
        print("\nYou can now run the webcam classifier with:")
        print("python webcam_fruit_classifier.py")
    else:
        print("✗ Some tests failed. Please check the model file and dependencies.")

if __name__ == "__main__":
    main() 