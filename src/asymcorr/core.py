import numpy as np
from scipy.stats import spearmanr, norm


class CorrelationUncertainty:
    """
    Compute Spearman correlation under measurement uncertainty using:

    - Monte Carlo perturbation sampling
    - Bootstrap resampling
    - Composite (MC + bootstrap) sampling
    """

    def __init__(self, x, y, xerr=None, yerr=None, random_state=None, nan_policy="propagate"):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.xerr = xerr
        self.yerr = yerr
        self.nan_policy = nan_policy
        self.rng = np.random.default_rng(random_state)
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate shapes and convert errors to 2xN arrays."""

        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

        self.xerr = self._validate_error(self.xerr, len(self.x))
        self.yerr = self._validate_error(self.yerr, len(self.y))

        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

        if self.xerr.shape != (2, len(self.x)):
            raise ValueError("xerr must have shape (2, len(x))")

        if self.yerr.shape != (2, len(self.y)):
            raise ValueError("yerr must have shape (2, len(y))")

    def _validate_error(self, err, n):
        """Convert error input into a (2, n) array."""
        if err is None:
            return np.zeros((2, n))

        err = np.asarray(err)
        if err.ndim == 1:  # symmetric error provided
            return np.vstack([err, err])

        return err

    def split_normal(self, mu, sigma_left, sigma_right, size=1):
        """
        Sample from a split (asymmetric) normal distribution.
        Left and right std devs determine which side is used.
        """
        mu = np.asarray(mu)
        sigma_left = np.asarray(sigma_left)
        sigma_right = np.asarray(sigma_right)

        # Safe elementwise division
        denom = sigma_left + sigma_right
        p_left = np.divide(sigma_left, denom, out=np.full_like(denom, 0.5, dtype=float), where=denom > 0)

        u = self.rng.uniform(0, 1, size=size)

        return np.where(
            u < p_left,
            self.rng.normal(loc=mu, scale=sigma_left, size=size),
            self.rng.normal(loc=mu, scale=sigma_right, size=size),
        )

    def prepare_samples_mc(self, n, indices=None):
        """Prepare Monte Carlo perturbed samples for x and y."""

        if indices is not None:
            x = self.x[indices]
            y = self.y[indices]
            xerr = self.xerr[:, indices]
            yerr = self.yerr[:, indices]
        else:
            x = self.x
            y = self.y
            xerr = self.xerr
            yerr = self.yerr

        x_samples = self.split_normal(x, xerr[0], xerr[1], size=(n, len(x)))
        y_samples = self.split_normal(y, yerr[0], yerr[1], size=(n, len(y)))
        return x_samples, y_samples

    # ----------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------
    def perturbation(self, n=10000):
        """
        Monte Carlo perturbation sampling.
        Returns arrays of rho and p values.
        """
        x_samples, y_samples = self.prepare_samples_mc(n)

        rhos = np.empty(n)
        pvals = np.empty(n)

        for i in range(n):
            rhos[i], pvals[i] = spearmanr(x_samples[i], y_samples[i], nan_policy=self.nan_policy)

        return rhos, pvals

    def bootstrap(self, n=10000):
        """
        Standard bootstrap sampling of (x, y) pairs.
        """
        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))

        rhos = np.empty(n)
        pvals = np.empty(n)

        for i in range(n):
            rhos[i], pvals[i] = spearmanr(
                self.x[indices[i]],
                self.y[indices[i]],
                nan_policy=self.nan_policy,
            )

        return rhos, pvals

    def composite(self, n=10000):
        """
        Composite method:
        bootstrap indices + Monte Carlo perturbation for each bootstrap sample.
        """

        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))

        rhos = np.empty(n)
        pvals = np.empty(n)

        for i, idx in enumerate(indices):
            x_s, y_s = self.prepare_samples_mc(1, indices=idx)
            x_s = x_s.flatten()
            y_s = y_s.flatten()
            rhos[i], pvals[i] = spearmanr(x_s, y_s, nan_policy=self.nan_policy)

        return rhos, pvals

    def compare_methods(self, n=10000, print_summary=True):
        """
        Compare all three methods + a standard calculation without uncertainty.
        Returns a dictionary of results or/and prints the summary.
        """
        results = {}

        rho, pval = spearmanr(self.x, self.y, nan_policy=self.nan_policy)
        results["standard"] = {rho, pval}
        rhos, pvals = self.perturbation(n)
        results["perturbation"] = self.summarise(rhos, pvals)
        rhos, pvals = self.bootstrap(n)
        results["bootstrap"] = self.summarise(rhos, pvals)
        rhos, pvals = self.composite(n)
        results["composite"] = self.summarise(rhos, pvals)

        if print_summary:
            rho, pval = results["standard"]
            pval = f"{pval:.2e}" if pval < 0.001 else f"{pval:.3f}"
            print(f"Standard method: {rho:.2f} (p={pval})")
            print(f"---" * 5)
            for method, summary in results.items():
                if method == "standard":
                    continue
                print(method.capitalize())
                self.print_summary(summary)
                print(f"---" * 5)

    @staticmethod
    def summarise(rhos, pvals, sigma=1, significance_level=0.05):
        """
        Summarise correlation results with median, std of rho and C.I. of p-values and significance fraction of p<0.05.
        """
        sigma = norm.sf(sigma)
        return {
            "rho_median": np.median(rhos),
            "rho_std": np.std(rhos),
            "rho_ci": (
                np.percentile(rhos, sigma * 100),  # 15.9th percentile
                np.percentile(rhos, (1 - sigma) * 100),
            ),
            "pval_median": np.median(pvals),
            "significant_fraction": np.sum(pvals < significance_level) / len(pvals),
        }

    @staticmethod
    def print_summary(summary):
        """
        Print summary dictionary in a readable format.
        """
        rho_median = f'Rho median: {summary["rho_median"]:.2f} ± {summary["rho_std"]:.2f}'
        cis = f'CI: ({summary["rho_ci"][0]:.2f}, {summary["rho_ci"][1]:.2f})'
        pval_median = summary["pval_median"]
        if pval_median < 0.001:
            pval_str = f"P-value median: {pval_median:.2e}"
        else:
            pval_str = f"P-value median: {pval_median:.3f}"
        signif_frac = f'Significant fraction (p < 0.05): {summary["significant_fraction"]:.2%}'

        print(rho_median, cis, pval_str, signif_frac, sep="\n")
