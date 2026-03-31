# initialization...

# %matplotlib inline

import pandas as pd                   # the panda—the python panel data analysis—library
import numpy as np                    # the numpy—the numerical python—library
import matplotlib.pyplot as plt       # pyplot from matplotlib—the python plotting interface module within the python matlab-style plotting library
import random as rnd                  # the standard random-number generation package
from scipy.optimize import curve_fit  # for more sophisticated and flexible functional forms for exploratory data analysis


# read in my dataframe of guesstimates and guesses for finger exercises
# on the longest-run structure of human economic history

import pandas as pd                   # the panda—the python panel data analysis—library


human_history_df = pd.read_feather("data/human_history_df.feather") # read in my dataframe

human_history_df.head(3) # check to see if everything is as expected

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def estimate_alpha_sigma(human_history_df = human_history_df, 
                         initial_year=-5000, final_year=200,
                         plot_start=-13000, plot_end=2025,
                         anchor_year=200, show_plot=False,
                         show_results=True
):
    """
    1. Estimate g_t = alpha * P_t^sigma on [initial_year, final_year] via curve_fit.
    2. Plot actual tech_growth vs fitted/extrapolated from plot_start to plot_end.
    3. Plot actual tech_index H vs fitted/extrapolated H_hat (log scale).
    """

    # --- 1. Estimation subset ---
    mask_est = (
        (human_history_df["year"] >= initial_year) &
        (human_history_df["year"] <= final_year)
    )
    df_est = human_history_df.loc[
        mask_est, ["year", "population", "tech_growth"]
    ].copy()
    df_est = df_est.dropna(subset=["population", "tech_growth"])
    df_est = df_est[df_est["population"] > 0].copy()

    def g_model(P, alpha, sigma):
        return alpha * np.power(P, sigma)

    popt, pcov = curve_fit(
        g_model,
        df_est["population"].values,
        df_est["tech_growth"].values,
        p0=[1e-4, 0.25],
        bounds=([0, 0], [1, 2])
    )
    alpha_hat, sigma_hat = popt
    se_alpha, se_sigma = np.sqrt(np.diag(pcov))

    if show_results:
        print(f"Estimation window: {initial_year} to {final_year}")
        print(f"  alpha_hat = {alpha_hat:.6e}  (SE {se_alpha:.2e})")
        print(f"  sigma_hat = {sigma_hat:.4f}       (SE {se_sigma:.4f})")
        print(f"  N obs     = {len(df_est)}")

    # --- 2. Full dataframe for plotting ---
    mask_all = (
        (human_history_df["year"] >= plot_start) &
        (human_history_df["year"] <= plot_end)
    )
    df = human_history_df.loc[
        mask_all, ["year", "population", "tech_index", "tech_growth"]
    ].copy()
    df = df[df["population"] > 0].sort_values("year").reset_index(drop=True)

    df["g_hat"] = alpha_hat * np.power(df["population"].values, sigma_hat)

    # --- 3. Integrate g_hat → H_hat, anchored ---
    i0 = df.index[df["year"] == anchor_year][0]
    H0 = df.loc[i0, "tech_index"]
    H_hat = np.full(len(df), np.nan)
    H_hat[i0] = H0

    for i in range(i0 + 1, len(df)):
        dt = df.loc[i, "year"] - df.loc[i - 1, "year"]
        H_hat[i] = H_hat[i - 1] * np.exp(df.loc[i, "g_hat"] * dt)

    for i in range(i0 - 1, -1, -1):
        dt = df.loc[i + 1, "year"] - df.loc[i, "year"]
        H_hat[i] = H_hat[i + 1] / np.exp(df.loc[i + 1, "g_hat"] * dt)

    df["H_hat"] = H_hat

    # --- 4. Masks for fitted vs extrapolated ---
    in_sample = (df["year"] >= initial_year) & (df["year"] <= final_year)
    before_sample = df["year"] < initial_year
    after_sample = df["year"] > final_year

    # --- 5. Plot tech_growth: actual vs fitted/extrapolated ---
    if show_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        ax = axes[0]
        ax.plot(df["year"], df["tech_growth"],
            'ko-', markersize=5, linewidth=1.5, label="Actual tech_growth", zorder=3)
        ax.plot(df.loc[in_sample, "year"], df.loc[in_sample, "g_hat"],
            'b--', linewidth=2, label=f"Fitted ({initial_year} to {final_year})")
        if before_sample.any():
            ax.plot(df.loc[before_sample, "year"], df.loc[before_sample, "g_hat"],
                'r:', linewidth=2, label=f"Extrapolated (before {initial_year})")
        if after_sample.any():
            ax.plot(df.loc[after_sample, "year"], df.loc[after_sample, "g_hat"],
                'r--', linewidth=2, label=f"Extrapolated (after {final_year})")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_ylabel("Tech growth rate g")
        ax.set_title(f"Tech Growth Rate:  α̂={alpha_hat:.3e},  σ̂={sigma_hat:.3f}"
                 f"  (est. {initial_year} to {final_year})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- 6. Plot tech level H: actual vs fitted/extrapolated ---
        ax = axes[1]
        valid = (df["tech_index"] > 0) & (df["H_hat"] > 0)
        ax.semilogy(df.loc[valid, "year"], df.loc[valid, "tech_index"],
                'ko-', markersize=5, linewidth=1.5, label="Actual H", zorder=3)
        fit_valid = valid & in_sample
        ax.semilogy(df.loc[fit_valid, "year"], df.loc[fit_valid, "H_hat"],
                'b--', linewidth=2, label="Fitted H_hat")
        ext_before = valid & before_sample
        if ext_before.any():
            ax.semilogy(df.loc[ext_before, "year"], df.loc[ext_before, "H_hat"],
                    'r:', linewidth=2, label="Extrapolated H_hat (before)")
        ext_after = valid & after_sample
        if ext_after.any():
            ax.semilogy(df.loc[ext_after, "year"], df.loc[ext_after, "H_hat"],
                    'r--', linewidth=2, label="Extrapolated H_hat (after)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Tech Index H (log scale)")
        ax.set_title("Tech Level: Actual vs Model")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return alpha_hat, sigma_hat, df


# def estimate_alpha_sigma(initial_year=-5000, final_year=200, plot_start=-13000, plot_end=200, anchor_year=200):

results = estimate_alpha_sigma(initial_year=-5000, final_year=200, 
                               plot_start=-13000, plot_end=1500, 
                               anchor_year=200, show_plot=True)

# Loop over specified initial and final years, collecting (initial_year, final_year, alpha_hat, sigma_hat)

results = []

initial_years = [-13000, -8000, -5000, -3000]
final_years   = [-600, 200, 800, 1300, 1600, 1775]

for iy in initial_years:
    for fy in final_years:
        try:
            alpha_hat, sigma_hat, df_est = estimate_alpha_sigma(
                human_history_df=human_history_df,
                initial_year=iy,
                final_year=fy,
                show_results=False
            )
            results.append({
                "initial_year": iy,
                "final_year": fy,
                "alpha_hat": alpha_hat,
                "sigma_hat": sigma_hat
            })
        except Exception as e:
            print(f"Skipping initial_year={iy}, final_year={fy}: {e}")

alpha_sigma_table = pd.DataFrame(results, columns=["initial_year", "final_year", "alpha_hat", "sigma_hat"])
alpha_sigma_table


# sanity check to see if we are in fact fitting the curve to the data

import random

# assume alpha_sigma_table already exists and has columns initial_year, final_year
# and estimate_alpha_sigma has the signature used in your plotting version:
# estimate_alpha_sigma(initial_year=..., final_year=..., plot_start=..., plot_end=..., anchor_year=..., show_plot=...)

# 1. Pick a random row
row = alpha_sigma_table.sample(1, random_state=None).iloc[0]
iy = int(row["initial_year"])
fy = int(row["final_year"])

print("Re-running estimation for:")
print(f"  initial_year = {iy}")
print(f"  final_year   = {fy}")

# 2. Call the plotting estimator on that window
_ = estimate_alpha_sigma(
    human_history_df=human_history_df,
    initial_year=iy,
    final_year=fy,
    plot_start=-13000,   # or whatever you prefer
    plot_end=2025,
    anchor_year=200,
    show_plot=True
)


human_history_df


























