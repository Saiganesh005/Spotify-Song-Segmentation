"""
=========================================================
Spotify Genre Segmentation
Model Evaluation Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)

os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


class ModelEvaluation:

    def __init__(self):

        self.results = {}

    # =====================================================
    # Save Plot
    # =====================================================

    def save_plot(self, filename):

        plt.tight_layout()

        plt.savefig(
            f"outputs/images/{filename}",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(f"Saved : outputs/images/{filename}")

    # =====================================================
    # Evaluate Clustering Model
    # =====================================================

    def evaluate_model(
        self,
        X,
        labels,
        model_name="Model"
    ):

        silhouette = silhouette_score(X, labels)

        calinski = calinski_harabasz_score(
            X,
            labels
        )

        davies = davies_bouldin_score(
            X,
            labels
        )

        self.results[model_name] = {

            "Silhouette Score": silhouette,

            "Calinski-Harabasz Score": calinski,

            "Davies-Bouldin Score": davies

        }

        print("=" * 60)
        print(model_name)
        print("=" * 60)

        print(f"Silhouette Score        : {silhouette:.4f}")
        print(f"Calinski-Harabasz Score : {calinski:.4f}")
        print(f"Davies-Bouldin Score    : {davies:.4f}")

        return self.results[model_name]

    # =====================================================
    # Silhouette Plot
    # =====================================================

    def silhouette_plot(
        self,
        X,
        labels
    ):

        values = silhouette_samples(
            X,
            labels
        )

        plt.figure(figsize=(8,5))

        plt.hist(
            values,
            bins=30
        )

        plt.xlabel("Silhouette Score")

        plt.ylabel("Frequency")

        plt.title("Silhouette Score Distribution")

        self.save_plot(
            "27_silhouette_distribution.png"
        )

    # =====================================================
    # Metric Comparison Plot
    # =====================================================

    def comparison_plot(self):

        if len(self.results) == 0:

            print("No evaluation results available.")

            return

        df = pd.DataFrame(self.results).T

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16,5)
        )

        sns.barplot(
            x=df.index,
            y=df["Silhouette Score"],
            ax=axes[0]
        )

        axes[0].set_title("Silhouette Score")

        sns.barplot(
            x=df.index,
            y=df["Calinski-Harabasz Score"],
            ax=axes[1]
        )

        axes[1].set_title("Calinski-Harabasz")

        sns.barplot(
            x=df.index,
            y=df["Davies-Bouldin Score"],
            ax=axes[2]
        )

        axes[2].set_title("Davies-Bouldin")

        self.save_plot(
            "28_model_comparison.png"
        )

    # =====================================================
    # Radar Chart Comparison
    # =====================================================

    def radar_chart(self):

        if len(self.results) == 0:

            return

        df = pd.DataFrame(self.results).T

        normalized = (

            df - df.min()

        ) / (

            df.max() - df.min() + 1e-8

        )

        labels = normalized.columns

        angles = np.linspace(
            0,
            2*np.pi,
            len(labels),
            endpoint=False
        )

        angles = np.concatenate(
            (
                angles,
                [angles[0]]
            )
        )

        plt.figure(figsize=(7,7))

        ax = plt.subplot(
            111,
            polar=True
        )

        for model in normalized.index:

            values = normalized.loc[
                model
            ].tolist()

            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=2,
                label=model
            )

            ax.fill(
                angles,
                values,
                alpha=0.2
            )

        ax.set_xticks(
            angles[:-1]
        )

        ax.set_xticklabels(labels)

        plt.title("Normalized Metric Comparison")

        plt.legend()

        self.save_plot(
            "29_radar_comparison.png"
        )

    # =====================================================
    # Save Results
    # =====================================================

    def save_results(self):

        if len(self.results) == 0:

            return

        df = pd.DataFrame(self.results).T

        df.to_csv(
            "outputs/model_evaluation.csv"
        )

        print(
            "Saved : outputs/model_evaluation.csv"
        )

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        if len(self.results) == 0:

            return

        df = pd.DataFrame(self.results).T

        print("=" * 70)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 70)

        print(df)

    # =====================================================
    # Complete Evaluation Pipeline
    # =====================================================

    def evaluate_pipeline(

        self,

        ml_features,

        ml_labels,

        dl_features,

        dl_labels

    ):

        self.evaluate_model(

            ml_features,

            ml_labels,

            "KMeans"

        )

        self.evaluate_model(

            dl_features,

            dl_labels,

            "Deep Learning"

        )

        self.silhouette_plot(

            ml_features,

            ml_labels

        )

        self.comparison_plot()

        self.radar_chart()

        self.save_results()

        self.summary()

        print("\nEvaluation Completed Successfully.")
