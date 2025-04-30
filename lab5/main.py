import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

def chi2_test(
    sample,
    alpha=0.05,
    k=10,
    out_dir='report/data',
    table_fname='chi2_table.tex',
    plot_fname='chi2_plot.png'
):
    os.makedirs(out_dir, exist_ok=True)
    data = np.asarray(sample)
    n = data.size
    mu_hat = data.mean()
    sigma_hat = data.std(ddof=0)
    intervals = np.linspace(data.min(), data.max(), k + 1)
    observed_freq, _ = np.histogram(data, bins=intervals)
    probs = np.diff(norm.cdf(intervals, loc=mu_hat, scale=sigma_hat))
    expected_freq = n * probs
    chi2_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()
    df = k - 1 - 2
    chi2_crit = chi2.ppf(1 - alpha, df)

    table = pd.DataFrame({
        "Интервал": [ f"$[{intervals[i]:.2f},\\ {intervals[i+1]:.2f})$" for i in range(k)],
        "$n_i$": observed_freq,
        "$n p_i$": np.round(expected_freq, 2),
        "$\\frac{(n_i - n p_i)^2}{n p_i}$": np.round(((observed_freq - expected_freq) ** 2 / expected_freq), 2)
    })

    table_path = os.path.join(out_dir, table_fname)
    table.to_latex(
        table_path,
        index=False,
        column_format='lccc',
        caption=(
            r"Таблица расчёта статистики $\chi^2$ "
            r"для проверки нормальности выборки "
            rf"$n={n},\ \alpha={alpha}$"
        ),
        label=r"tab:chi2_test",
        escape=False,
        bold_rows=True,
        position='H',
        float_format="%.2f"
    )

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=intervals, density=True, alpha=0.5)
    x = np.linspace(data.min(), data.max(), 200)
    plt.plot(x, norm.pdf(x, mu_hat, sigma_hat), 'r')
    plt.title(
        rf"$n={n},\ \hat{{\mu}}={mu_hat:.2f},\ \hat{{\sigma}}={sigma_hat:.2f},"
        rf"\ \chi^2={chi2_stat:.2f}\ (\chi^2_{{crit}}={chi2_crit:.2f})$"
    )
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plot_path = os.path.join(out_dir, plot_fname)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return {
        'mu_hat': mu_hat,
        'sigma_hat': sigma_hat,
        'chi2_stat': chi2_stat,
        'chi2_crit': chi2_crit,
        'df': df,
        'table_path': table_path,
        'plot_path': plot_path
    }

np.random.seed(0)
result = chi2_test(
    np.random.normal(0, 1, 20),
    alpha=0.05,
    k=10,
    table_fname='normal_20.tex',
    plot_fname='normal_20.png'
)



result = chi2_test(
    np.random.normal(0, 1, 100),
    alpha=0.05,
    k=10,
    table_fname='normal_100.tex',
    plot_fname='normal_100.png'
)


result = chi2_test(
    np.random.uniform(-np.sqrt(3), np.sqrt(3), 20),
    alpha=0.05,
    k=10,
    table_fname='uniform_20.tex',
    plot_fname='uniform_20.png'
)



result = chi2_test(
    np.random.uniform(-np.sqrt(3), np.sqrt(3), 100),
    alpha=0.05,
    k=10,
    table_fname='uniform_100.tex',
    plot_fname='uniform_100.png'
)


import subprocess
report_directory = 'report'
os.chdir('report')
subprocess.run(['pdflatex', 'main.tex'], check=True)
