import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('_mpl-gallery-nogrid')
# print(plt.style.available)

def analyze_distributions(sample_sizes):
    distributions = {
        'Нормальнго распределения': lambda n: np.random.normal(0, 1, n),
        'распределения Коши': lambda n: np.random.standard_cauchy(size=n),
        'распределения Пуассона': lambda n: np.random.poisson(10, n),
        'Равномерного распределения': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    }

    results = {key: {size: {} for size in sample_sizes} for key in distributions.keys()}
    titles = [ 'Размер выборки', 'Количество выбросов']

    results = {name: {size: {} for size in sample_sizes} for name in distributions.keys()}

    for name, dist_func in distributions.items():
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        
        for size in sample_sizes:
            data_array = dist_func(size)

            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
            num_outliers = outliers.size 
            results[name][size]['Количество выбросов'] = num_outliers
            results[name][size]['Размер выборки'] = size
            
            data_to_plot.append(data_array)

        plt.boxplot(data_to_plot, vert=True, tick_labels=sample_sizes)
        plt.title(f'Бокс-плот {name}')
        plt.xlabel('Размер выборки')
        plt.savefig(f'./отчет/data/boxplot_{name}.png')
        plt.close()

    for key in distributions.keys():        
        with open(f'./отчет/data/{key}.tex', 'w') as f:

            f.write(r'\begin{figure}[h]' + '\n')
            f.write(r'    \centering' + '\n')
            f.write(r'    \includegraphics[width=1.0\textwidth]{./data/boxplot_' + key + r'.png}' + '\n')
            f.write(r'    \caption{бокс-плоты Тьюки для ' + key + r'}' + '\n')
            f.write(r'    \label{fig:' + key + r'}' + '\n')
            f.write(r'\end{figure}' + '\n')
        
            f.write('\\begin{table}[h]\n')
            f.write('   \\centering\n')
            f.write('   \\begin{tabular}{|' + '|'.join(['c'] * len(titles)) + '|}\n')
            f.write('       \\hline\n')
            results_lines = ' & '.join([f'{title}' for title in titles])
            f.write('        ' + results_lines + ' \\\\\n')
            f.write('       \\hline\n')

            for size in sample_sizes:
                f.write(f'        ')
                results_lines = []

                for title in titles:
                    val = results[key][size][title]
                    results_lines.append(f'${val}$')

                f.write(' & '.join(results_lines) + '\\\\\n')
                f.write('        \\hline\n')

            f.write('   \\end{tabular}' + '\n')
            f.write(f'   \\caption{{Результаты для {key}}}\n')
            f.write('\\end{table}' + '\n')


sample_sizes = [20, 100, 1000]
analyze_distributions(sample_sizes)
