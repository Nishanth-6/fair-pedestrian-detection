import pandas as pd

def dataset_summary(path="runs/detect/predict/predictions.csv"):
    df = pd.read_csv(path)
    print("Total detections:", len(df))
    print("Classes present:", df['class_name'].value_counts())

if __name__ == "__main__":
    dataset_summary()