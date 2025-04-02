import matplotlib.pyplot as plt
import numpy as np
import math
import os

def normal(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def cauchy(x, loc=0, scale=1):
    return 1 / (np.pi * scale * (1 + ((x - loc) / scale)**2))

def poisson_pmf(k, lam=10):
    k = [int(i) for i in k]
    values = np.zeros_like(k, dtype=float)
    for i in range(len(k)):
        values[i] = (lam ** k[i] * np.exp(-lam)) / math.factorial(k[i])
    
    return values

def uniform(x, a=-np.sqrt(3), b=np.sqrt(3)):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)


def plot_distributions(distribution_name, data, func, bins, x_min, x_max):
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)

    hist_counts, bin_edges = np.histogram(data, bins=bins, range=(x_min, x_max))
    ylim1 = max(np.mean(hist_counts) * 2, np.max(hist_counts) * 1.1)
    ylim2 = ylim1 * bins / (np.sum((data >= x_min) & (data <= x_max))* (x_max - x_min))

    if ylim2 < max(y):
        ylim2 = max(y) * 1.1
        ylim1 = ylim2 * (np.sum((data >= x_min) & (data <= x_max))* (x_max - x_min)) / bins

    fig, ax1 = plt.subplots(figsize=(6, 6))

    ax1.hist(data, bins=bins, density=False, color='blue', label='Гистограмма', range=(x_min, x_max))
    ax1.set_xlabel('Значение')
    ax1.set_ylabel('Частота', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, ylim1)

    ax1.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, y, color='red', label='Плотность распределения', alpha=0.5)
    ax2.set_ylabel('Плотность', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, ylim2)

    plt.savefig(f'./отчет/data_1/{distribution_name}_{len(data)}.png', dpi=300)
    # plt.show()

    plt.close()

def generate_latex_figure(val, sizes):
    with open(f'./отчет/data_1/{val}.tex', 'w') as f:
        f.write('\\begin{figure}[H]' + '\n')
        f.write('   \\centering' + '\n')

        for size in sizes:
            f.write('   \\begin{subfigure}[b]{0.47\\textwidth}' + '\n')
            f.write('       \\centering' + '\n')
            f.write(f'       \\includegraphics[width=\\linewidth]{{data_1/{val}_{size}.png}}' + '\n')
            f.write(f'       \\caption{{Гистограмма и график плотности при $n = {size}$}}' + '\n')
            f.write('   \\end{subfigure}' + '\n')
            f.write('   \\hfill' + '\n')

        f.write('   \\caption{Гистограммы и графики плотности для ' + val + '}' + '\n')
        f.write('\\end{figure}' + '\n\n\n')


def main():
    sizes = [10, 50, 1000]

    for size in sizes:
        normal_data = np.random.normal(0, 1, size)
        bins = int(1 + 2.5 * math.log(size))
        dx = max(np.max(normal_data), - np.min(normal_data) )
        x_min = - dx - 1
        x_max = dx + 1
        plot_distributions('нормального распределения', normal_data, normal, bins, x_min, x_max)

    for size in sizes:
        cauchy_data = np.random.standard_cauchy(size)
        bins = int(1 + 10 * math.sqrt(size))
        dx = max(np.max(cauchy_data), - np.min(cauchy_data) )
        x_min = - dx
        x_max = dx
        plot_distributions('распределения Коши', cauchy_data, cauchy, bins, x_min, x_max)


    for size in sizes:
        poisson_data = np.random.poisson(10, size)
        x_min = 0
        x_max = max(poisson_data) + 1  
        bins = x_max - x_min
        plot_distributions('распределения Пуассона', poisson_data, poisson_pmf, bins, x_min, x_max)


    for size in sizes:
        uniform_data = np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
        bins = int(1 + 3 * math.log(size))
        x_min = -np.sqrt(3)
        x_max = np.sqrt(3)
        plot_distributions('равномерного распределения', uniform_data, uniform, bins, x_min, x_max)



    generate_latex_figure('нормального распределения', sizes)
    generate_latex_figure('распределения Коши', sizes)
    generate_latex_figure('распределения Пуассона', sizes)
    generate_latex_figure('равномерного распределения', sizes)
    
    
if __name__ == "__main__":
    main()
