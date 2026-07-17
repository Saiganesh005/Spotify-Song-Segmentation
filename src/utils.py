"""
=========================================================
Spotify Genre Segmentation
Utility Functions
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

from src.import_libraries import *
from src.config import *


# =====================================================
# Create Directory
# =====================================================

def create_directory(path):

    """
    Create directory if it doesn't exist.
    """

    os.makedirs(path, exist_ok=True)


# =====================================================
# Create Project Directories
# =====================================================

def create_project_directories():

    directories = [

        DATA_DIR,

        RAW_DATA_DIR,

        PROCESSED_DATA_DIR,

        MODEL_DIR,

        OUTPUT_DIR,

        IMAGE_DIR,

        REPORT_DIR,

        PREDICTION_DIR

    ]

    for directory in directories:

        create_directory(directory)

    print("Project directories verified.")


# =====================================================
# Save DataFrame
# =====================================================

def save_dataframe(df, path):

    df.to_csv(path, index=False)

    print(f"Saved : {path}")


# =====================================================
# Load DataFrame
# =====================================================

def load_dataframe(path):

    return pd.read_csv(path)


# =====================================================
# Save Model
# =====================================================

def save_model(model, path):

    joblib.dump(model, path)

    print(f"Model Saved : {path}")


# =====================================================
# Load Model
# =====================================================

def load_model(path):

    model = joblib.load(path)

    print(f"Model Loaded : {path}")

    return model


# =====================================================
# Save Scaler
# =====================================================

def save_scaler(scaler):

    joblib.dump(

        scaler,

        SCALER_FILE

    )

    print("Scaler Saved.")


# =====================================================
# Load Scaler
# =====================================================

def load_scaler():

    scaler = joblib.load(

        SCALER_FILE

    )

    print("Scaler Loaded.")

    return scaler


# =====================================================
# Save Plot
# =====================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            IMAGE_DIR,

            filename

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"Plot Saved : {filename}")


# =====================================================
# Save Figure
# =====================================================

def save_current_figure(filename):

    save_plot(filename)


# =====================================================
# PCA Transformation
# =====================================================

def perform_pca(data, components=2):

    pca = PCA(

        n_components=components,

        random_state=RANDOM_STATE

    )

    transformed = pca.fit_transform(data)

    return transformed


# =====================================================
# Random Sample
# =====================================================

def random_sample(dataframe, size=500):

    if len(dataframe) <= size:

        return dataframe

    return dataframe.sample(

        size,

        random_state=RANDOM_STATE

    )


# =====================================================
# Print Heading
# =====================================================

def print_heading(title):

    print("\n")

    print("=" * 60)

    print(title.upper())

    print("=" * 60)


# =====================================================
# Dataset Information
# =====================================================

def dataset_information(df):

    print_heading("Dataset Information")

    print(df.info())

    print()

    print("Shape :", df.shape)


# =====================================================
# Memory Usage
# =====================================================

def memory_usage(df):

    usage = df.memory_usage(

        deep=True

    ).sum() / (1024 ** 2)

    print(f"Memory Usage : {usage:.2f} MB")

    return usage


# =====================================================
# Timer
# =====================================================

def start_timer():

    return time.time()


def stop_timer(start_time):

    elapsed = time.time() - start_time

    print(f"Execution Time : {elapsed:.2f} seconds")

    return elapsed


# =====================================================
# Banner
# =====================================================

def project_banner():

    print("=" * 60)

    print("SPOTIFY GENRE SEGMENTATION")

    print("Machine Learning Project")

    print("=" * 60)


# =====================================================
# Check File Exists
# =====================================================

def file_exists(path):

    return os.path.exists(path)


# =====================================================
# Seed Everything
# =====================================================

def seed_everything(seed=RANDOM_STATE):

    np.random.seed(seed)

    random.seed(seed)

    try:

        tf.random.set_seed(seed)

    except:

        pass

    print(f"Random Seed : {seed}")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    project_banner()

    create_project_directories()

    seed_everything()

    print("Utilities Ready.")
