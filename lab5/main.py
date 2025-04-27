import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import os

def chi2_normality_test(sample, alpha=0.05, k=10, table_filename="report/data/table.tex", hist_filename="report/data/histogram.png"):
    n = len(sample)

    mu_hat = np.mean(sample)
    sigma_hat = np.sqrt(np.mean((sample - mu_hat)**2))

    quantiles = np.linspace(0, 1, k+1)
    bin_edges = stats.norm.ppf(quantiles, loc=mu_hat, scale=sigma_hat)
    bin_edges[0] = sample.min() - 1
    bin_edges[-1] = sample.max() + 1
    observed_counts, _ = np.histogram(sample, bins=bin_edges)
    expected_counts = np.full(k, n / k)
    chi2_components = (observed_counts - expected_counts)**2 / expected_counts
    chi2_statistic = np.sum(chi2_components)
    df = k - 2 - 1
    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    table_data = {
        'Интервал': [str(i+1) for i in range(k)],
        'Нижняя граница': [round(bin_edges[i], 3) for i in range(k)],
        'Верхняя граница': [round(bin_edges[i+1], 3) for i in range(k)],
        'Ожидаемая частота': [round(expected_counts[i], 3) for i in range(k)],
        'Наблюдаемая частота': observed_counts.tolist(),
        'Вклад в χ²': [round(chi2_components[i], 3) for i in range(k)]
    }
    df_table = pd.DataFrame(table_data)
    test_result = "Не отвергаем H0" if chi2_statistic < chi2_critical else "Отвергаем H0"
    print(f"Оценка параметров: μ̂ = {mu_hat:.4f}, σ̂ = {sigma_hat:.4f}\n")
    print("Таблица вычислений для критерия χ²:")
    print(df_table.to_string(index=False))
    print("\nРезультаты проверки гипотезы:")
    print(f"Статистика χ² = {chi2_statistic:.3f}")
    print(f"Критическое значение χ² при df = {df} и α = {alpha}: {chi2_critical:.3f}")
    print(f"Вывод: {test_result} (распределение {'соответствует' if test_result == 'Не отвергаем H0' else 'не соответствует'} N(μ̂, σ̂)).")
    
    x_values = np.linspace(sample.min(), sample.max(), 1000)
    pdf_values = stats.norm.pdf(x_values, loc=mu_hat, scale=sigma_hat)
    plt.figure(figsize=(8, 5))
    plt.hist(sample, bins=bin_edges, density=True, label='Выборка')
    plt.plot(x_values, pdf_values, label='Оцененная плотность N(μ̂, σ̂)')
    plt.xlabel('x')
    plt.ylabel('Плотность')
    plt.title('Гистограмма выборки и оцененная нормальная плотность')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(os.path.dirname(hist_filename)):
        os.makedirs(os.path.dirname(hist_filename))
    plt.savefig(hist_filename)
    plt.show()

    if not os.path.exists(os.path.dirname(table_filename)):
        os.makedirs(os.path.dirname(table_filename))
    latex_table = df_table.to_latex(index=False, escape=False, caption='', label='tab:uniform', column_format='l|c', position='h!')
    with open(table_filename, "w") as f:
        f.write(latex_table)
        
    result = {
        'mu_hat': mu_hat,
        'sigma_hat': sigma_hat,
        'chi2_statistic': chi2_statistic,
        'chi2_critical': chi2_critical,
        'degrees_of_freedom': df,
        'table': df_table,
        'test_result': test_result
    }
    return result

if __name__ == '__main__':
    np.random.seed(0)
    
    n = 100
    sample_normal = np.random.normal(loc=0, scale=1, size=n)
    print("Проверка выборки из нормального распределения (n=100):")
    res_normal = chi2_normality_test(sample_normal, table_filename="report/data/normal_table.tex", hist_filename="report/data/normal_histogram.png")
    
    sample_uniform_20 = np.random.uniform(low=-1, high=1, size=20)
    print("\nПроверка выборки из равномерного распределения (n=20):")
    res_uniform_20 = chi2_normality_test(sample_uniform_20, table_filename="report/data/uniform20_table.tex", hist_filename="report/data/uniform20_histogram.png")

    sample_uniform_100 = np.random.uniform(low=-1, high=1, size=100)
    print("\nПроверка выборки из равномерного распределения (n=100):")
    res_uniform_100 = chi2_normality_test(sample_uniform_100, table_filename="report/data/uniform100_table.tex", hist_filename="report/data/uniform100_histogram.png")
    