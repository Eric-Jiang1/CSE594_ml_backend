#!/usr/bin/env python3
"""
Helper script to fix model file compatibility issues.
Run this on Python 3.12 (the version the model was saved with) to create
a compatible version for Python 3.9.

Usage:
    python3.12 fix_model.py
"""

import joblib
import sys

def fix_model(input_path="risk_model.pkl", output_path="risk_model.pkl"):
    """Load model on Python 3.12 and save without SHAP explainer."""
    print(f"Loading model from {input_path}...")
    try:
        bundle = joblib.load(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    if not isinstance(bundle, dict):
        print("Model is not a dictionary bundle. Nothing to fix.")
        return False
    
    print("Model loaded successfully!")
    print(f"Keys in bundle: {list(bundle.keys())}")
    
    # Remove SHAP-related items that cause compatibility issues
    items_to_remove = ["shap_explainer", "shap_background", "raw_tree_model"]
    removed = []
    for key in items_to_remove:
        if key in bundle:
            removed.append(key)
            del bundle[key]
    
    if removed:
        print(f"Removed SHAP-related items: {removed}")
        print("(These aren't needed for predictions)")
    
    # Keep only what's needed for predictions
    essential_keys = ["model", "encoder", "imputer", "features", "feature_cols"]
    print(f"\nEssential components:")
    for key in essential_keys:
        if key in bundle:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} (missing)")
    
    print(f"\nSaving fixed model to {output_path}...")
    try:
        joblib.dump(bundle, output_path)
        print("✓ Model saved successfully!")
        print(f"\nYou can now use this model on Python 3.9.")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

if __name__ == "__main__":
    if sys.version_info < (3, 12):
        print("WARNING: This script should be run on Python 3.12")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    fix_model()


