import numpy as np
import matplotlib.pyplot as plt

def compute_statistics(data):
    mean_x = np.mean(data)
    median_x = np.median(data)
    z_1_4 = np.percentile(data, 25)
    z_3_4 = np.percentile(data, 75)
    z_Q = (z_1_4 + z_3_4) / 2
    return mean_x, median_x, z_Q

def generate_data(dist, size):
    if dist == 1:
        return np.random.normal(0, 1, size)
    elif dist == 2:
        return np.random.standard_cauchy(size)
    elif dist == 3:
        return np.random.poisson(10, size)
    elif dist == 4:
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
    
def format_scientific(value, n=1):
    parts = "{:.10e}".format(value).split("e")

    mantissa = float(parts[0])
    exponent = int(parts[1])
    if n == 0:
        mantissa = int(np.round(mantissa))
        if exponent == 1:
            return f"{mantissa*10}"
        if exponent != 0:
            return f"{mantissa} \\cdot 10^{{{exponent}}}"
        return f"{mantissa}"
    
    if n == -1:
        return f"{"-" if mantissa < 0 else ""}10^{{{exponent}}}"
    
    if exponent == 1:
        return f"{mantissa*10:.{n}f}"
    
    formatted_mantissa = f"{mantissa:.{n}f}"
    if exponent != 0:
        return f"{formatted_mantissa} \\cdot 10^{{{exponent}}}"
    else:
        return f"{formatted_mantissa}"

def main():
    sizes = [10, 100, 1000]
    num_iterations = 1_000

    distributions = {
    1: 'нормального распределения',
    2: 'распределения Коши',
    3: 'распределения Пуассона',
    4: 'равномерного распределения'
    }
    titles = ["E(mean x)", 'D(mean x)', 'E(med x)', 'D(med x)', 'E(z_Q)', 'D(z_Q)']

    mean_x = np.zeros(num_iterations)
    median_x = np.zeros(num_iterations)
    z_Q = np.zeros(num_iterations)
    results = {dist: {} for dist in distributions}

    for dist in distributions.keys():
        for size in sizes:
            for i in range(num_iterations):
                data = generate_data(dist, size)

                mean_x_i, median_x_i, z_Q_i = compute_statistics(data)
                mean_x[i] = mean_x_i
                median_x[i] = median_x_i
                z_Q[i] = z_Q_i

            mean_x_avg = np.mean(mean_x)
            median_x_avg = np.mean(median_x)
            z_Q_avg = np.mean(z_Q)

            E_mean_x = np.mean(mean_x**2)
            E_median_x = np.mean(median_x**2)
            E_z_Q = np.mean(z_Q**2)
            
            D_mean_x = E_mean_x - mean_x_avg**2
            D_median_x = E_median_x - median_x_avg**2
            D_z_Q = E_z_Q - z_Q_avg**2

            values = [mean_x_avg, D_mean_x, median_x_avg, D_median_x, z_Q_avg, D_z_Q]
            results[dist][size] = dict(zip(titles, values))

    # for key, val in distributions.items():
    #     print(f'Распределение: {val}')
    #     for size in sizes:
    #         print(f'  Размер выборки: {size}')
    #         for title in titles:
    #             print(f'    {title}: {results[key][size][title]}')


    # for key, val in distributions.items():
    #     for idx, title in enumerate(titles):
    #         # plt.subplot(2, 3, idx + 1)
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(sizes, [results[key][size][title] for size in sizes], marker='+')
    #         # plt.title(f'{title} для распределения {val}')
    #         plt.xlabel('Размер выборки')
    #         plt.ylabel(f'Значние {title}')
    #         plt.xticks(sizes)
    #         plt.xscale('log')
    #         plt.grid()
    #         plt.savefig(f'./отчет/data_2/{title}_{val}.png', dpi=300)
    #         plt.close() 
    
        # plt.savefig(f'{output_dir}/{val}.png', dpi=300)
        # plt.show()
        
    for key, val in distributions.items():
        with open(f'./отчет/data_2/{val}.tex', 'w') as f:
            f.write('\\begin{table}[h]\n')
            f.write('   \\centering\n')
            f.write(f'   \\caption{{Результаты для {val}}}\n')
            f.write('   \\begin{tabular}{|c|' + '|'.join(['c'] * len(titles)) + '|}\n')
            f.write('       \\hline\n')
            results_lines = ''
            for title in titles:
                if results_lines:
                    results_lines += ' & '
                results_lines += f'${title}$'  
            f.write('       & ' + results_lines + ' \\\\\n')
            f.write('       \\hline\n')

            for size in sizes:
                f.write(f'        Размер выборки: {size} & ')
                results_lines = []

                for i in range(len(titles)//2):
                    val = results[key][size][titles[2*i]]
                    dis = results[key][size][titles[2*i+1]]
                    if np.sqrt(dis) > abs(val - np.round(val)):
                        results_lines.append(f'${int(np.round(int(val/dis) * dis))}$ & ${format_scientific(dis , 0)}$ &')
                    else:
                        order_of_magnitude = int(np.log10(dis))
                        n = max(0, -order_of_magnitude)
                        results_lines.append(f'${format_scientific(val , n)}$ & ${format_scientific(dis, 0)}$ &')

                f.write(''.join(results_lines)[:-2] + '\\\\\n')
                f.write('        \\hline\n')

            f.write('   \\end{tabular}\n')
            f.write('\\end{table}\n')



if __name__ == "__main__":
    main()
