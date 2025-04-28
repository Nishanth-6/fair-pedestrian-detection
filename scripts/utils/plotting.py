import matplotlib.pyplot as plt

def plot_height_distribution(df):
    person_df = df[df['class_name'] == 'person'].copy()
    person_df['height'] = person_df['ymax'] - person_df['ymin']

    plt.hist(person_df['height'], bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.xlabel("Bounding Box Height (pixels)")
    plt.ylabel("Count")
    plt.title("Person Bounding Box Height Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()