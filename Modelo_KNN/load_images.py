import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

IMG_SIZE = 64  # padroniza tamanho

def load_dataset(base_path, max_per_class=None):
    data = []
    labels = []
    
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        count = 0
        
        for img_name in os.listdir(class_path):
            if max_per_class and count >= max_per_class:
                break
            
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            
            data.append(img)
            labels.append(class_name)
            count += 1
    
    return np.array(data), np.array(labels)