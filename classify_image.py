#!/usr/bin/env python3
"""
Simple script to classify a single image using the fruit classifier.
This is a wrapper around the main webcam_fruit_classifier.py script.
"""

import sys
import os
from webcam_fruit_classifier import FruitClassifier

def main():
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <image_path>")
        print("Example: python classify_image.py my_fruit_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    # Initialize classifier
    print("Loading fruit classifier...")
    classifier = FruitClassifier()
    
    if classifier.model is None:
        print("Failed to load model. Please check the model file.")
        sys.exit(1)
    
    # Classify the image
    print(f"\nClassifying image: {image_path}")
    result = classifier.classify_image(image_path, show_image=True, save_result=True)
    
    if result[0]:
        print(f"\nClassification completed successfully!")
        print(f"Result: {result[0]} (confidence: {result[1]:.2%})")
    else:
        print("Classification failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 