import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ----------- DATA PATH -----------
data_path = "data/cell_images"

# ----------- PREPROCESS FUNCTION -----------
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img

# ----------- SEGMENTATION FUNCTION -----------
def segment_image(img):
    img_uint8 = (img * 255).astype('uint8')
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ----------- FEATURE EXTRACTION FUNCTION -----------
def extract_features(segmented, processed):
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [0]*8  # updated size
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # -------- Basic features --------
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Mean intensity inside cell
    mask = segmented > 0
    mean_intensity = processed[mask].mean() if mask.any() else 0
    
    # -------- Existing features --------
    
    # Bounding box → aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0
    
    # Compactness (shape irregularity)
    compactness = (perimeter ** 2) / (area + 1e-5)
    
    # -------- NEW FEATURES (HIGH IMPACT) --------
    
    # Extent (how filled bounding box is)
    extent = area / (w * h + 1e-5)
    
    # Solidity (convexity / concavity)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-5)
    
    # Equivalent diameter (stable size measure)
    equiv_diameter = np.sqrt(4 * area / np.pi)
    
    return [
        area,
        perimeter,
        mean_intensity,
        aspect_ratio,
        compactness,
        #extent,
        solidity
        #equiv_diameter
    ]

# ----------- BUILD DATASET FUNCTION -----------
def build_dataset(data_path):
    X = []
    y = []
    
    classes = ["Parasitized", "Uninfected"]
    
    for label, cls in enumerate(classes):
        class_path = os.path.join(data_path, cls)
        
        for img_name in os.listdir(class_path)[:300]:  # limit for speed
            img_path = os.path.join(class_path, img_name)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            processed = preprocess_image(img)
            segmented = segment_image(processed)
            features = extract_features(segmented, processed)
            
            X.append(features)
            y.append(label)
    
    return np.array(X), np.array(y)

# ----------- SAMPLE IMAGE VISUALIZATION -----------

sample_path = os.path.join(data_path, "Parasitized")
img_name = os.listdir(sample_path)[0]

img_path = os.path.join(sample_path, img_name)
img = cv2.imread(img_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#print("Original shape:", img_rgb.shape)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

processed = preprocess_image(img)

#print("Processed shape:", processed.shape)

plt.imshow(processed, cmap='gray')
plt.title("Processed Image")
plt.axis("off")
plt.show()

segmented = segment_image(processed)

plt.imshow(segmented, cmap='gray')
plt.title("Segmented Image")
plt.axis("off")
plt.show()

features = extract_features(segmented, processed)

#print("\nExtracted Features:")
#print(f"Area: {features[0]}")
#print(f"Perimeter: {features[1]}")
#print(f"Mean Intensity: {features[2]}")

# ----------- BUILD DATASET -----------

print("\nBuilding dataset...")
X, y = build_dataset(data_path)

print("Dataset shape:", X.shape)

# ----------- TRAIN TEST SPLIT -----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ----------- TRAIN MODEL -----------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
model.fit(X_train, y_train)

# ----------- EVALUATE -----------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")