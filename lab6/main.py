import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


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

# def fmt_interval(name, iv):
#     a = iv[0]
#     b = iv[1]
#     return f"${name}\\in[{a:.2f},\\,{b:.2f}]$"

# np.random.seed(0)
# samples = {20: np.random.normal(0, 1, 20), 100:np.random.normal(0, 1, 100)}

# rows_normal = []
# rows_asym   = []

# for n, sample in samples.items():
#     mL, mU, sL, sU = ci_normal(sample)
#     rows_normal.append({
#         "n": n,
#         "$m$": [mL, mU],
#         "$\\sigma$": [sL, sU]
#     })

#     mL2, mU2, sL2, sU2 = ci_asymptotic(sample)
#     rows_asym.append({
#         "n": n,
#         "$m$": [mL2, mU2],
#         "$\\sigma$": [sL2, sU2]
#     })

# df_normal = pd.DataFrame(rows_normal)
# df_asym    = pd.DataFrame(rows_asym)

# os.makedirs(out_dir, exist_ok=True)

# df_normal.to_latex(
#     os.path.join(out_dir, "normal_intervals.tex"),
#     index=False,
#     column_format="|c|c|c|",
#     caption="Доверительные интервалы для параметров нормального распределения",
#     label="tab:normal_intervals",
#     escape=False,
#     position="H",
#     formatters={ "$m$": lambda x: fmt_interval('m', x),
#                  "$\\sigma$": lambda x: fmt_interval('\\sigma', x) }
# )

# df_asym.to_latex(
#     os.path.join(out_dir, "asym_intervals.tex"),
#     index=False,
#     column_format="|c|c|c|",
#     caption="Доверительные интервалы для параметров произвольного распределения (асимптотический подход)",
#     label="tab:asym_intervals",
#     escape=False,
#     position="H",
#     formatters={ "$m$": lambda x: fmt_interval('m', x),
#                  "$\\sigma$": lambda x: fmt_interval('\\sigma', x) }
# )

np.random.seed(0)
samples = {20: np.random.normal(0, 1, 20), 100:np.random.normal(0, 1, 100)}

rows_normal = []
rows_asym = []
alpha = 0.05
for n, sample in samples.items():
    mL, mU, sL, sU = ci_normal(sample, alpha)
    rows_normal.append({
        "n": n,
        "$m$": f"{mL:.2f} < $m$ < {mU:.2f}",
        "$\\sigma$": f"{sL:.2f} < $\\sigma$ < {sU:.2f}"
    })
    mL2, mU2, sL2, sU2 = ci_asymptotic(sample, alpha)
    rows_asym.append({
        "n": n,
        "$m$": f"{mL2:.2f} < $m$ < {mU2:.2f}",
        "$\\sigma$": f"{sL2:.2f} < $\\sigma$ < {sU2:.2f}"
    })

df_normal = pd.DataFrame(rows_normal)
df_asym = pd.DataFrame(rows_asym)

df_normal.to_latex(
    os.path.join(out_dir, "normal_intervals.tex"),
    index=False,
    column_format="|c|c|c|",
    caption=f"Доверительные интервалы для параметров нормального распределения,  $\\alpha={alpha}$",
    label="tab:normal_intervals",
    escape=False,
    position="H"
)

df_asym.to_latex(
    os.path.join(out_dir, "asym_intervals.tex"),
    index=False,
    column_format="|c|c|c|",
    caption=f"Доверительные интервалы для параметров произвольного распределения (асимптотический подход),  $\\alpha={alpha}$",
    label="tab:asym_intervals",
    escape=False,
    position="H"
)
