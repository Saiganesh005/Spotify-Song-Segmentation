"""
Spotify Genre Segmentation
Training Script
"""

from src.machine_learning import train_model
from src.deep_learning import train_autoencoder

def main():

    print("=" * 50)
    print("Training Machine Learning Model")
    print("=" * 50)

    train_model()

    print()

    print("=" * 50)
    print("Training Deep Learning Model")
    print("=" * 50)

    train_autoencoder()

    print("\nTraining Completed Successfully.")

if __name__ == "__main__":
    main()
