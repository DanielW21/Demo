import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

import wandb


class MetricsCalculator:
    def __init__(self, threshold: float, iter: int = 0) -> None:
        self.y_test: np.ndarray = None
        self.y_test_pred: np.ndarray = None
        self.metrics: Dict[str, float] = {}

        self.y_tests: List[np.ndarray] = []
        self.y_test_preds: List[np.ndarray] = []

        self.threshold: float = threshold
        self.iteration: int = (
            iter  
        )

    def calculate_metrics(self, y_test, y_test_pred, update=True) -> None:
        """
        Calculate and store the following metrics:
        - R2 Score
        - Explained Variance
        - Spearman's Rank Correlation
        - Precision
        - Recall
        - Mean Absolute Error
        - Mean Squared Error
        """
        self.y_test = y_test
        self.y_test_pred = y_test_pred

        if update:
            self.y_tests.append(y_test)
            self.y_test_preds.append(y_test_pred)

        r2: float = r2_score(self.y_test, self.y_test_pred)
        spearman_corr: float = spearmanr(self.y_test, self.y_test_pred).correlation
        mse: float = mean_squared_error(self.y_test, self.y_test_pred)
        mae: float = mean_absolute_error(self.y_test, self.y_test_pred)
        explain_var: float = explained_variance_score(self.y_test, self.y_test_pred)

        precision, recall = self.calculate_precision_recall()

        self.metrics = {
            "R2 Score": r2,
            "Explained Variance": explain_var,
            "Spearman's Rank Correlation": spearman_corr,
            "Precision": precision,
            "Recall": recall,
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
        }

    def calculate_precision_recall(self, threshold=None) -> tuple[float, float]:
        threshold = (
            threshold if threshold is not None else self.threshold
        )  # Default Threshold

        binary_true = (self.y_test >= threshold).astype(int)
        binary_predictions = (self.y_test_pred >= threshold).astype(int)

        TP = np.sum((binary_predictions == 1) & (binary_true == 1))
        FP = np.sum((binary_predictions == 1) & (binary_true == 0))
        FN = np.sum((binary_predictions == 0) & (binary_true == 1))
        TN = np.sum((binary_predictions == 0) & (binary_true == 0))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        return precision, recall

    def log_metrics(self) -> None:
        """
        Update Log Dictionary
        """
        if self.iteration:
            print("\nIteration: ", self.iteration, "\n")

        for metric_name, metric_value in self.metrics.items():
            print(f"{metric_name}: {metric_value}")

    def log_avg_metrics(self, config) -> None:
        """
        When using k-fold cross-validation, log the average metrics of all folds Using MetricCalculator y_preds avg
        Takes List of y_preds and the y_test true values
        Calculates each individual fold metrics and then logs the average of all folds
        """
        k_metrics = {
            "R2 Score": [],
            "Explained Variance": [],
            "Spearman's Rank Correlation": [],
            "Precision": [],
            "Recall": [],
            "Mean Absolute Error": [],
            "Mean Squared Error": [],
        }

        n = len(self.y_tests)

        for idx in range(n):
            cur_y_test = self.y_tests[idx]
            cur_y_test_pred = self.y_test_preds[idx]
            self.calculate_metrics(
                cur_y_test, cur_y_test_pred, update=False
            )  

            for metric_name, metric_value in self.metrics.items():
                k_metrics[metric_name].append(metric_value)

        print("\n Final Averages: \n")
        for metric in k_metrics:
            avg_metric = np.mean(k_metrics[metric])

            if len(k_metrics[metric]) > 1:
                print(f"{metric}: {avg_metric} ± {np.std(k_metrics[metric])}") 
            else:
                print(f"{metric}: {avg_metric}")  

            wandb.summary[metric] = avg_metric

        model_name = config["model"]
        path = config["local"]["path"]
        output_path = f"{path}/{model_name}_data.csv"

        print("\n   Data Set: ", config["data"]["k_fold_split"])
        print("   Split Type: ", config["method"])
        print("   Model: ", model_name)

        counter = 1
        while os.path.exists(output_path):
            output_path = f"{path}/{model_name}_data_{counter}.csv"
            counter += 1

        entries = max(
            len(y_test) for y_test in self.y_tests
        )  

        print("Lengths of all folds:")
        for idx, (y_test, y_pred) in enumerate(zip(self.y_tests, self.y_test_preds)):
            print(
                f"Fold {idx+1}: y_test length = {len(y_test)}, y_test_pred length = {len(y_pred)}"
            )

        df = pd.DataFrame()

        for idx in range(n):
            assert len(self.y_tests[idx]) == len(
                self.y_test_preds[idx]
            ), f"Length mismatch in fold {idx+1}"

            y_test_padded, y_pred_padded = [], []

            if len(self.y_tests[idx]) < entries:
                y_test_padded = np.concatenate(
                    [
                        self.y_tests[idx],
                        np.full((entries - len(self.y_tests[idx]),), float("nan")),
                    ]
                )
                y_pred_padded = np.concatenate(
                    [
                        self.y_test_preds[idx],
                        np.full((entries - len(self.y_test_preds[idx]),), float("nan")),
                    ]
                )

            else:
                y_test_padded, y_pred_padded = self.y_tests[idx], self.y_test_preds[idx]

            df[f"y_test_{idx+1}"], df[f"y_test_pred_{idx+1}"] = (
                y_test_padded,
                y_pred_padded,
            )

        df.to_csv(output_path, index=False)

        print(f"Data saved to {output_path}")

    def precision_recall(self, intervals=100):
        """
        Plots the precision, recall, and f1 score for different thresholds and logs it to WandB.

        Args:
            y_test (np.array): True labels
            y_preds (np.array): Predicted labels
            intervals (int): Number of intervals to divide the threshold range into
        """
      
        thresholds = np.linspace(self.y_test.min(), self.y_test.max(), intervals)
        precisions = []
        recalls = []
        f1_scores = []

        for thresh in thresholds:
            binarized_preds = (self.y_test_pred >= thresh).astype(int)
            binarized_true = (self.y_test >= thresh).astype(int)

            precision = precision_score(
                binarized_true, binarized_preds, zero_division=0
            )
            recall = recall_score(binarized_true, binarized_preds, zero_division=0)
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision", color="blue")
        plt.plot(thresholds, recalls, label="Recall", color="red")
        plt.plot(thresholds, f1_scores, label="F1 Score", color="green")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold vs Precision/Recall/F1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        wandb.log(
            {f"Threshold vs Precision/Recall/F1 | {self.iteration}": wandb.Image(plt)}
        )

        plt.close()

    def pr_curve(self, num_thresholds=100):
        """
        Plots the precision/recall curve for the model's predictions and logs it to WandB.
        """
        thresholds = np.linspace(0, 1, num_thresholds)

        precisions = []
        recalls = []

        for threshold in thresholds:
            precision, recall = self.calculate_precision_recall(threshold)
            precisions.append(precision)
            recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker=".", color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)

        wandb.log({"Precision-Recall Curve": wandb.Image(plt)})

        plt.close()

    def bargraph(self, true_train_labels=None, bins=50):
        """
        Plots and logs the distribution of predicted vs true labels to WandB.

        Args:
            y_pred (np.array): Predicted labels
            y_test (np.array): True labels
            bins (int): Number of bins for the histogram
            wandb_key (str): Key to log the image to wandb
        """

        counts_pred, bin_edges = np.histogram(self.y_test_pred, bins=bins, density=True)
        counts_true, _ = np.histogram(self.y_test, bins=bin_edges, density=True)
        counts_train_true, _ = np.histogram(
            true_train_labels, bins=bin_edges, density=True
        )

        plt.figure(figsize=(8, 6))
        plt.bar(
            bin_edges[:-1],
            counts_train_true,
            width=np.diff(bin_edges),
            edgecolor="black",
            alpha=0.6,
            label="Train Labels",
            color="khaki",
        )

        plt.bar(
            bin_edges[:-1],
            counts_pred,
            width=np.diff(bin_edges),
            edgecolor="black",
            alpha=0.6,
            label="Test Prediction Labels",
        )
        plt.bar(
            bin_edges[:-1],
            counts_true,
            width=np.diff(bin_edges),
            edgecolor="black",
            alpha=0.4,
            label="Test Labels",
            color="lightgreen",
        )

        plt.xlabel("Label")
        plt.ylabel("Frequency")
        plt.title("Label Distribution: Predicted vs True")
        plt.legend()
        plt.grid(True)

        wandb.log(
            {
                f"Label Distribution (Predicted vs True)  | {self.iteration}": wandb.Image(
                    plt
                )
            }
        )

        plt.close()

    def scatterplot(self, y_data_list):
        """
        Plots a scatter plot of predicted vs. true values with different colors for
        train, validation, and test sets (if available). Skips empty entries.

        Args:
            y_data_list (list of lists): List of [y_test, y_test_pred] pairs.
                                        Expected order: [[y_train, y_train_pred], [y_valid, y_valid_pred], [y_test, y_test_pred]]
                                        Use empty list ([]) to skip a dataset.
        """
        plt.figure(figsize=(6, 6))

        colors = ["lightgreen", "gold", "darkred"]
        labels = ["Train", "Valid", "Test"]

        legend_entries = []
        min_vals, max_vals = [], []

        for idx, item in enumerate(y_data_list):
            if not item or len(item) != 2:
                continue 

            y_true, y_pred = item
            sns.scatterplot(
                x=y_true,
                y=y_pred,
                alpha=0.6,
                color=colors[idx],
                label=labels[idx],
                s=12,
            )

            r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
            legend_entries.append(f"{labels[idx]} ($r^2$={r2:.2f})")

            min_vals.extend([y_true.min(), y_pred.min()])
            max_vals.extend([y_true.max(), y_pred.max()])

        if min_vals and max_vals:
            min_val = min(min_vals)
            max_val = max(max_vals)
            plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5)

        plt.xlabel("True Labels")
        plt.ylabel("Model Prediction")
        plt.title("Model Correlation")

        if legend_entries:
            plt.legend(labels=legend_entries, loc="upper left", frameon=True)

        wandb.log({f"Predicted vs True Labels | {self.iteration}": wandb.Image(plt)})

        plt.close()

    def top_k(self):
        """
        Graphs the size of the top-k' true values that are in the top-k predictions for all values of k
        """
        n = len(self.y_test)

        ordered_predictions = np.argsort(self.y_test_pred)[::-1]
        ordered_trues = np.argsort(self.y_test)[::-1]

        scores = [float("inf") for _ in range(n)]


        cur_points = set()

        cur_k = 0

        for i in range(n):
            cur_points.add(ordered_predictions[i])

            while cur_k < n and ordered_trues[cur_k] in cur_points:
                cur_k += 1

            scores[i] = cur_k

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n + 1), scores, color="blue")
        plt.xlabel("Number of Predictions")
        plt.ylabel("Top-k")
        plt.title("Top-k")
        plt.grid(True)

        wandb.log({f"Top-k | {self.iteration}": wandb.Image(plt)})

        plt.close()

    def percent_top_k(self, quantity=False):
        """
        Graphs the % of the top-k true values that are in the top-k predictions for all values of k
        """
        
        n = len(self.y_test)

        ordered_predictions = np.argsort(self.y_test_pred)[::-1]
        ordered_trues = np.argsort(self.y_test)[::-1]

        scores = [0 for _ in range(n)]

        top_k_true = set()
        top_k_preds = set()

        for i in range(n):
            top_k_true.add(ordered_trues[i])
            top_k_preds.add(ordered_predictions[i])

            scores[i] = len(top_k_true.intersection(top_k_preds))

            if not quantity:
                scores[i] = scores[i] / (i + 1) * 100

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n + 1), scores, color="blue")
        plt.xlabel("Number of Top-k Predictions")

        if quantity:
            plt.ylabel("Quantity of True Top-k Predictions")
            plt.ylim(0, n)
        else:
            plt.ylabel("% Predictions in True Top-k")
            plt.ylim(0, 100)

        plt.title("Union of Top-k Predictions and Top-k True Values")
        plt.grid(True)

        wandb.log({f"Top-k Union | {self.iteration}": wandb.Image(plt)})

        plt.close()
