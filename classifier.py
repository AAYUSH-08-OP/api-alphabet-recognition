import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

clf = LogisticRegression()

X = np.load("image.npz")["arr-0"]
y = np.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts)

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 3500, test_size = 500)


def get_prediction(image):
    im_pil = Image.open(image)
    img_bw = im_pil.convert("L")
    img_bw_resized = img_bw.resize((22, 30), Image.ANTIALIAS)
    pixel_filters = 20
    min_pixel = np.percentile(img_bw_resized, pixel_filters)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]