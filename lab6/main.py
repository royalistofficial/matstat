import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


out_dir = "report/results"
Path(out_dir).mkdir(exist_ok=True)

def ci_normal(sample, alpha=0.05):
    n = len(sample)
    xbar = np.mean(sample)
    s = np.std(sample, ddof=1)
    t = stats.t.ppf(1-alpha/2, df=n-1)
    m_lower = xbar - t * s / np.sqrt(n)
    m_upper = xbar + t * s / np.sqrt(n)
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1-alpha/2, df=n-1)
    sigma_lower = s * np.sqrt((n-1)/chi2_upper)
    sigma_upper = s * np.sqrt((n-1)/chi2_lower)
    return m_lower, m_upper, sigma_lower, sigma_upper

def ci_asymptotic(sample, alpha=0.05):
    n = len(sample)
    xbar = np.mean(sample)
    s = np.std(sample, ddof=1)
    z = stats.norm.ppf(1-alpha/2)
    m_lower = xbar - z * s / np.sqrt(n)
    m_upper = xbar + z * s / np.sqrt(n)
    e = stats.kurtosis(sample, fisher=True, bias=False)
    U = z * np.sqrt((e + 2)/n)
    sigma_lower = s * (1 + U) ** (-0.5)
    sigma_upper = s * (1 - U) ** (-0.5)
    return m_lower, m_upper, sigma_lower, sigma_upper

# np.random.seed(0)
samples = {20: np.random.normal(0, 1, 20), 
           100:np.random.normal(0, 1, 100)}

alphas = (0.05, 0.5, 0.95) 
results = {
    n: {"alpha": [], "mL": [], "mU": [], "sL": [], "sU": [], 
        "mL2": [], "mU2": [], "sL2": [], "sU2": []}
    for n in samples
}


for alpha in alphas:
    rows_normal = []
    rows_asym = []

    for n, sample in samples.items():
        mL, mU, sL, sU = ci_normal(sample, alpha)
        rows_normal.append({
            "n": n,
            "Доверительный интервал для $m$": f"${mL:.2f} < m < {mU:.2f}$",
            "Доверительный интервал для $\\sigma$": f"${sL:.2f} < \\sigma < {sU:.2f}$"
        })
        
        mL2, mU2, sL2, sU2 = ci_asymptotic(sample, alpha)
        rows_asym.append({
            "n": n,
            "Доверительный интервал для $m$": f"${mL2:.2f} < m < {mU2:.2f}$",
            "Доверительный интервал для $\\sigma$": f"${sL2:.2f} < \\sigma < {sU2:.2f}$"
        })
        res = results[n]
        res["alpha"].append(alpha)
        res["mL"].append(mL)
        res["mU"].append(mU)
        res["sL"].append(sL)
        res["sU"].append(sU)
        res["mL2"].append(mL2)
        res["mU2"].append(mU2)
        res["sL2"].append(sL2)
        res["sU2"].append(sU2)


    df_normal = pd.DataFrame(rows_normal)
    df_asym = pd.DataFrame(rows_asym)

    df_normal.to_latex(
        os.path.join(out_dir, f"normal_intervals_alpha_{round(alpha, 2)}.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для нормального распределения при $\\alpha = {round(alpha, 2)}$",
        label="tab:normal_intervals_alpha",
        escape=False,
        position="H"
    )

    df_asym.to_latex(
        os.path.join(out_dir, f"asym_intervals_alpha_{round(alpha, 2)}.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для произвольного распределения (асимптотический подход) при $\\alpha = {round(alpha, 2)}$",
        label="tab:asym_intervals_alpha",
        escape=False,
        position="H"
    )

