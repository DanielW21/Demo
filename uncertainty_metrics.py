import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

import wandb
from configs.configure import ConfigDict


class UncertaintyEval:
    def __init__(self, config):
        self.config = config
        self.y_true = []
        self.y_pred = []
        self.y_var = []
        self.y_std = []
        self.iter = 1

    def set_data(self, y_true, y_pred, y_var, iter):
        """Set the true values, predicted values, and standard deviations"""
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_var = y_var
        self.y_std = np.sqrt(y_var)
        self.iter = iter

    def negative_log_likelihood(self):
        return -np.mean(norm.logpdf(self.y_true, loc=self.y_pred, scale=self.y_std))

    def picp(self, confidence=0.95):
        z_score = norm.ppf((1 + confidence) / 2)
        lower_bound = self.y_pred - z_score * self.y_std
        upper_bound = self.y_pred + z_score * self.y_std
        within_interval = np.logical_and(
            self.y_true >= lower_bound, self.y_true <= upper_bound
        )
        return np.mean(within_interval)

    def mpiw(self, confidence=0.95):
        z_score = norm.ppf((1 + confidence) / 2)
        interval_widths = 2 * z_score * self.y_std
        return np.mean(interval_widths)

    def crps(self):
        z = (self.y_true - self.y_pred) / self.y_std
        return np.mean(
            self.y_std
            * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        )

    def calibration_curve(self, n_bins=10):
        z = (self.y_true - self.y_pred) / self.y_std
        quantiles = np.linspace(0, 1, n_bins + 1)[1:]
        expected_probs = quantiles
        observed_probs = []

        for q in quantiles:
            threshold = norm.ppf(q)
            observed_prob = np.mean(z <= threshold)
            observed_probs.append(observed_prob)

        return expected_probs, np.array(observed_probs)

    def plot_calibration_curve(self, n_bins=10):
        """
        Plot the calibration curve (reliability diagram) for the model predictions.
        The calibration curve shows the relationship between the predicted probabilities
        and the observed probabilities.
        Args:
            n_bins (int): Number of bins to use for the calibration curve.
        """

        expected, observed = self.calibration_curve(n_bins)
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(expected, observed, "ro-", label="Model")
        plt.xlabel("Expected cumulative probability")
        plt.ylabel("Observed cumulative probability")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True)

        wandb.log({f"Calibration Curve | {self.iter}": wandb.Image(plt)})

    def uncertainty_graph(self, num_samples_per_bucket=20):
        """
        Plot the predicted labels against the true labels with error bars representing
        the uncertainty (standard deviation) of the predictions. The data is divided into
        bins based on the true labels, and a specified number of samples are randomly
        selected from each bin for plotting.

        Args:
            num_samples_per_bucket (int): Number of samples to plot from each bin.
        """
        bins = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
        bin_indices = np.digitize(self.y_true, bins) - 1 
        unique_bins = np.unique(bin_indices)

        plt.figure(figsize=(7, 7))

        for bin_idx in unique_bins[1:-1]:
            indices_in_bin = np.where(bin_indices == bin_idx)[0]

            if len(indices_in_bin) > num_samples_per_bucket:
                sampled_indices = np.random.choice(
                    indices_in_bin, num_samples_per_bucket, replace=False
                )
            else:
                sampled_indices = indices_in_bin

            true_samples = self.y_true[sampled_indices]
            pred_samples = self.y_pred[sampled_indices]
            std_samples = self.y_std[sampled_indices]

            plt.errorbar(
                true_samples,
                pred_samples,
                yerr=std_samples,
                fmt="o",
                color="red",
                alpha=0.6,
                label="Predictions ± Std" if bin_idx == unique_bins[1] else "",
            )

        plt.xlabel("True Labels")
        plt.xlim(-3.2, 3.2)

        plt.ylabel("Predicted Labels")
        plt.title("Predicted Labels vs True Labels with Uncertainty")
        plt.legend()
        plt.grid(True)

        wandb.log(
            {f"True vs Predicted with Uncertainty | {self.iter}": wandb.Image(plt)}
        )

    def uncertainty_error_graph(self):
        """
        Plot both scatter and kernel density (KDE) plots to visualize the relationship
        between uncertainty (std dev) and absolute error.
        """
        uncertainty_error = np.abs(self.y_true - self.y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].scatter(self.y_std, uncertainty_error, alpha=0.5)
        axes[0].set_xlabel("Uncertainty (Standard Deviation)")
        axes[0].set_ylabel("Absolute Error")
        axes[0].set_title("Scatter: Uncertainty vs Absolute Error")
        axes[0].grid(True)

        sns.kdeplot(
            x=self.y_std, y=uncertainty_error, fill=True, thresh=0.05, ax=axes[1]
        )
        axes[1].set_xlabel("Uncertainty (Standard Deviation)")
        axes[1].set_ylabel("Absolute Error")
        axes[1].set_title("KDE: Uncertainty vs Absolute Error")
        axes[1].grid(True)

        wandb.log({f"Uncertainty vs Absolute Error | {self.iter}": wandb.Image(fig)})

        plt.close(fig)


if __name__ == "__main__":
    config = ConfigDict()
    wandb.init(**config["wandb"])
    wandb.config.update(config)

    np.random.seed(42)  
    y_true = np.random.normal(loc=0.0, scale=1.0, size=400) 
    y_pred = y_true + np.random.normal(loc=0.0, scale=1.0, size=len(y_true))
    y_var = np.random.uniform(0.1, 0.5, size=400)

    evaluator = UncertaintyEval(config=config)
    evaluator.set_data(y_true, y_pred, y_var, iter=1)

    nll = evaluator.negative_log_likelihood()
    picp = evaluator.picp()
    mpiw = evaluator.mpiw()
    crps = evaluator.crps()

    print("Negative Log Likelihood:", nll)
    print("Prediction Interval Coverage Probability (PICP):", picp)
    print("Mean Prediction Interval Width (MPIW):", mpiw)
    print("Continuous Ranked Probability Score (CRPS):", crps)

    wandb.log(
        {"Negative Log Likelihood": nll, "PICP": picp, "MPIW": mpiw, "CRPS": crps}
    )

    evaluator.plot_calibration_curve()

    evaluator.uncertainty_graph()

    wandb.finish()
