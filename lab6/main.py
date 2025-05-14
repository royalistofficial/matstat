import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
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

np.random.seed(0)
samples = {20: np.random.normal(0, 1, 20), 
           100:np.random.normal(0, 1, 100)}

alphas = [0.05] 


for alpha in alphas:
    rows_normal = []

    rows_normal_2 = []

    for n, sample in samples.items():
        mL, mU, sL, sU = ci_normal(sample, alpha)
        x_inn = [mL + sL, mU - sL]
        x_out = [mL - sU, mU + sU]
        plt.figure(figsize=(10, 6))
        plt.hist(sample, edgecolor='black', alpha=0.6)

        # plt.axvspan(mL, mU, color='red', alpha=0.3, label=f"Доверительный интервал для $m$: ${mL:.2f} < m < {mU:.2f}$")

        plt.axvspan(*x_out, color='red', alpha=0.1, label=f"$x_{{out}}$")
        plt.axvspan(*x_inn, color='green', alpha=0.3, label=f"$x_{{inn}}$")


        plt.title(f'Гистограмма с $x_{{inn}}$ и $x_{{out}}$ для $\\alpha = {round(alpha, 2)}$, n = {n}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        
        plt.legend() 

        plt.savefig(os.path.join(out_dir, f'histogram_with_CI_{n}_alpha_{round(alpha, 2)}.png'), dpi=300)

        # plt.show()
        plt.close()


        rows_normal.append({
            "n": n,
            "Доверительный интервал для $m$": f"${mL:.2f} < m < {mU:.2f}$",
            "Доверительный интервал для $\\sigma$": f"${sL:.2f} < \\sigma < {sU:.2f}$"
        })

        rows_normal_2.append({
            "n": n,
            "$x_{inn}$": f"[${x_inn[0]:.2f}, {x_inn[1]:.2f}$]",
            "$x_{out}$": f"[${x_out[0]:.2f}, {x_out[1]:.2f}$]",
        })
        
        


    df_normal = pd.DataFrame(rows_normal)

    df_normal.to_latex(
        os.path.join(out_dir, f"normal_intervals_alpha_{round(alpha, 2)}.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для нормального распределения при $\\alpha = {round(alpha, 2)}$",
        label="tab:normal_intervals_alpha",
        escape=False,
        position="H"
    )

    df_normal = pd.DataFrame(rows_normal_2)

    df_normal.to_latex(
        os.path.join(out_dir, f"normal_intervals_alpha_{round(alpha, 2)}_2.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для нормального распределения при $\\alpha = {round(alpha, 2)}$",
        label="tab:normal_intervals_alpha_2",
        escape=False,
        position="H"
    )

samples = {20: np.random.uniform(-np.sqrt(3), np.sqrt(3), 20), 
           100: np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)}


for alpha in alphas:
    rows_asym = []
    rows_asym_2 = []

    for n, sample in samples.items():

        mL, mU, sL, sU = ci_asymptotic(sample, alpha)

        x_inn = [mL + sL, mU - sL]
        x_out = [mL - sU, mU + sU]
        plt.figure(figsize=(10, 6))
        plt.hist(sample, edgecolor='black', alpha=0.6)

        # plt.axvspan(mL, mU, color='red', alpha=0.3, label=f"Доверительный интервал для $m$: ${mL:.2f} < m < {mU:.2f}$")

        plt.axvspan(*x_out, color='red', alpha=0.1, label=f"$x_{{out}}$")
        plt.axvspan(*x_inn, color='green', alpha=0.3, label=f"$x_{{inn}}$")

        plt.title(f'Гистограмма с $x_{{inn}}$ и $x_{{out}}$ для $\\alpha = {round(alpha, 2)}$, n = {n}')

        plt.xlabel('Значение')
        plt.ylabel('Частота')
        
        plt.legend() 

        plt.savefig(os.path.join(out_dir, f'histogram_with_CI_2_{n}_alpha_{round(alpha, 2)}.png'), dpi=300)

        # plt.show()
        plt.close()

        rows_asym.append({
            "n": n,
            "Доверительный интервал для $m$": f"${mL:.2f} < m < {mU:.2f}$",
            "Доверительный интервал для $\\sigma$": f"${sL:.2f} < \\sigma < {sU:.2f}$"
        })
        rows_asym_2.append({
            "n": n,
            "$x_{inn}$": f"[${x_inn[0]:.2f}, {x_inn[1]:.2f}$]",
            "$x_{out}$": f"[${x_out[0]:.2f}, {x_out[1]:.2f}$]",
        })
        

    df_asym = pd.DataFrame(rows_asym)

    df_asym.to_latex(
        os.path.join(out_dir, f"asym_intervals_alpha_{round(alpha, 2)}.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для произвольного распределения (асимптотический подход) при $\\alpha = {round(alpha, 2)}$",
        label="tab:asym_intervals_alpha",
        escape=False,
        position="H"
    )

    df_asym = pd.DataFrame(rows_asym_2)

    df_asym.to_latex(
        os.path.join(out_dir, f"asym_intervals_alpha_{round(alpha, 2)}_2.tex"),
        index=False,
        column_format="|c|c|c|c|",
        caption=f"Доверительные интервалы для произвольного распределения (асимптотический подход) при $\\alpha = {round(alpha, 2)}$",
        label="tab:asym_intervals_alpha_2",
        escape=False,
        position="H"
    )
    
import subprocess
os.chdir('report')
subprocess.run(['pdflatex', 'main.tex'], check=True)

