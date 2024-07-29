import os
import cv2
import numpy as np
import pandas as pd
import skimage
from skimage import feature
import pickle

# Load model SVM
svm_model_path = "models/ModelSVM.pkl"
with open(svm_model_path, 'rb') as f:
    svm_model = pickle.load(f)

# Load model Random Forest
rf_model_path = "models/ModelRF.pkl"
with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

mean_std = pd.DataFrame({
        'mean': [0.003599440429750442, 0.9960282021407117, 0.9982002797851247, 0.15406044900306307, 0.003599440429750442, 1127.6454545454546, 0.006531534239851955],
        'std_dev': [0.0025960962243579133, 0.0030038336356668863, 0.0012980481121789517, 0.09096476275595562, 0.0025960962243579133, 895.0442953198701, 0.005184072575086734]
    })

def extract_glcm_features(image_path):
    img = skimage.io.imread(image_path, as_gray=True)
    img = np.asarray(img, dtype="int32")

    g = skimage.feature.graycomatrix(img, [1], [np.pi/2], levels=img.max()+1, symmetric=False, normed=True)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    g = feature.graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_dissimilarity = feature.graycoprops(g, 'dissimilarity')[0][0]
    glcm_contrast = feature.graycoprops(g, 'contrast')[0][0]
    glcm_energy = feature.graycoprops(g, 'energy')[0][0]
    glcm_homogeneity = feature.graycoprops(g, 'homogeneity')[0][0]
    glcm_correlation = feature.graycoprops(g, 'correlation')[0][0]

    # Return a DataFrame with the GLCM features
    glcm_feature_df = pd.DataFrame({
        'contrast': [glcm_contrast],
        'energy': [glcm_energy],
        'homogeneity': [glcm_homogeneity],
        'correlation': [glcm_correlation],
        'dissimilarity': [glcm_dissimilarity]
    })

    return glcm_feature_df

def extract_jala_features(image_path):
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholding
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create mask
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Apply mask
    segmented = cv2.bitwise_and(image, mask)
    # Compute features
    num_white_pixels = np.sum(segmented == 255)
    area = gray.shape[0] * gray.shape[1]
    jala_density = num_white_pixels / area

    jala_feature_df = pd.DataFrame({
        'jumlah piksel jala': num_white_pixels,
        'kepadatan piksel jala': jala_density
    }, index=[0])

    return jala_feature_df

def combine_features(image_path):
    glcm_features = extract_glcm_features(image_path)
    jala_features = extract_jala_features(image_path)
    combined_features = pd.concat([glcm_features, jala_features], axis=1)
    
    return combined_features

def normalize_features(combined_features, mean_std):

    means = mean_std['mean'].values
    stds = mean_std['std_dev'].values

    normalized_features = (combined_features - means) / stds
    
    return normalized_features

def predict_rf(normalized_features):
    prediction = rf_model.predict(normalized_features)
    return prediction

def predict_svm(normalized_features):
    prediction = svm_model.predict(normalized_features)
    return prediction