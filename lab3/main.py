import os
import re
from math import log10, ceil, sqrt
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse

def generate_normal(n, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def generate_mixture(n, *wargs):
    mean1 = [0, 0]
    cov1 = [[1, 0.9], [0.9, 1]]
    
    mean2 = [0, 0]
    cov2 = [[10, -9], [-9, 10]]
    
    choices = np.random.choice([1, 2], size=n, p=[0.9, 0.1])
    
    samples = np.array([
        np.random.multivariate_normal(mean1, cov1) if choice == 1 else np.random.multivariate_normal(mean2, cov2)
        for choice in choices
    ])
    np.random.shuffle(samples)
    return samples

def plot_ellipse(ax, mean, cov, n_std=1.645, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    ell.set_alpha(0.2)

def plot_samples_and_ellipse(generator, sizes, params, output_dir):
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)
    
    mixture_components = {
        'main': {'cov': [[1, 0.9], [0.9, 1]], 'color': 'red'},
        'outlier': {'cov': [[10, -9], [-9, 10]], 'color': 'blue'}
    }

    for size in sizes:
        for param in params:
            data = generator(size, param)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(data[:,0], data[:,1], alpha=0.5, s=20)
            
            if param == 'mixture':
                for comp in mixture_components.values():
                    plot_ellipse(
                        ax, 
                        mean=[0,0], 
                        cov=comp['cov'], 
                        color=comp['color'],
                        linewidth=1.5
                    )
                filename = f'mixture_n{size}.png'
            else:
                cov = [[1, param], [param, 1]]
                plot_ellipse(ax, [0,0], cov, color='green')
                filename = f'normal_n{size}_rho{param}.png'.replace('.', '_')
            
            path = os.path.join(output_dir, filename)
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            image_paths.append(f'./img/{filename}')
    
    return image_paths

def run_experiment(generator, sizes, params, n_runs=1000, is_mixture=False):
    results = {}
    
    for size in sizes:
        for param in params:
            if is_mixture:
                key = (size, 'mixture')
                if key in results: 
                    continue
                print(f"Обработка смеси, размер={size}")
            else:
                key = (size, param)
                print(f"Обработка нормального распределения, размер={size}, rho={param}")
            
            pearson = np.zeros(n_runs)
            spearman = np.zeros(n_runs)
            quadratic = np.zeros(n_runs)
            
            for i in range(n_runs):
                if is_mixture:
                    data = generator(size)
                else:
                    data = generator(size, param)
                
                pearson[i] = stats.pearsonr(data[:, 0], data[:, 1])[0]
                spearman[i] = stats.spearmanr(data[:, 0], data[:, 1])[0]
                quadratic[i] = np.corrcoef(data[:, 0], data[:, 1])[0, 1]**2
            
            results[key] = {
                'pearson_mean': np.mean(pearson),
                'pearson_var': np.var(pearson),
                'spearman_mean': np.mean(spearman),
                'spearman_var': np.var(spearman),
                'quadratic_mean': np.mean(quadratic),
                'quadratic_var': np.var(quadratic)
            }
    
    return results

def save_to_latex(results, filename, dist_type, image_paths):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(r"\subsubsection{Графики}" + "\n")
        # Обработка изображений группами по 3 штуки
        for i in range(0, len(image_paths), 3):
            group = image_paths[i:i+3]
            f.write(r"\begin{figure}[H]" + "\n")
            f.write(r"\centering" + "\n")
            # Для каждой картинки в группе используем subfigure
            for img_path in group:
                # Пытаемся извлечь параметр n из имени файла, например "n20"
                match = re.search(r"n(\d+)", img_path)
                caption = f"$n={match.group(1)}$" if match else ""
                
                f.write(r"\begin{subfigure}[b]{0.3\textwidth}" + "\n")
                f.write(r"\centering" + "\n")
                f.write(fr"\includegraphics[width=\textwidth]{{{img_path}}}" + "\n")
                f.write(fr"\caption{{{caption}}}" + "\n")
                f.write(r"\end{subfigure}" + "\n")
                f.write(r"\quad" + "\n")
            # Общая подпись для группы изображений с уточнением типа распределения
            f.write(r"\caption{Примеры выборок для распределения " + dist_type + "}" + "\n")
            f.write(r"\end{figure}" + "\n\n")
        
        # Добавление таблиц с результатами
        f.write(r"\subsubsection{Результаты}" + "\n")
        if dist_type == 'normal':
            f.write(r"""
                \begin{tabular}{|c|c|c|c|c|c|c|c|}
                \hline
                Размер & $\rho$ & \multicolumn{2}{c|}{Пирсон} & \multicolumn{2}{c|}{Спирмен} & \multicolumn{2}{c|}{Квадратичный} \\ \hline
                & & Среднее & Дисперсия & Среднее & Дисперсия & Среднее & Дисперсия \\ \hline
                """)
                
            for (size, rho), res in results.items():
                n1 = max(int(ceil(-log10(res['pearson_var']))) , 0)
                n2 = max(int(ceil(-log10(res['spearman_var']))) , 0)
                n3 = max(int(ceil(-log10(res['quadratic_var']))), 0)
                f.write(f"{size} & {rho} & "
                        f"{res['pearson_mean']:.{n1}f} & {sp.latex(sp.Rational(res['pearson_var']).evalf(1))} & "
                        f"{res['spearman_mean']:.{n2}f} & {sp.latex(sp.Rational(res['spearman_var']).evalf(1))} & "
                        f"{res['quadratic_mean']:.{n3}f} & {sp.latex(sp.Rational(res['quadratic_var']).evalf(1))} \\\\ \\hline\n")
        else:
            f.write(r"""
                \begin{tabular}{|c|c|c|c|c|c|c|}
                \hline
                Размер & \multicolumn{2}{c|}{Пирсон} & \multicolumn{2}{c|}{Спирмен} & \multicolumn{2}{c|}{Квадратичный} \\ \hline
                & Среднее & Дисперсия & Среднее & Дисперсия & Среднее & Дисперсия \\ \hline
                """)
                
            for (size, rho), res in results.items():
                n1 = max(int(ceil(-log10(res['pearson_var']))) , 0)
                n2 = max(int(ceil(-log10(res['spearman_var']))) , 0)
                n3 = max(int(ceil(-log10(res['quadratic_var']))), 0)
                f.write(f"{size} & "
                        f"{res['pearson_mean']:.{n1}f} & {sp.latex(sp.Rational(res['pearson_var']).evalf(1))} & "
                        f"{res['spearman_mean']:.{n2}f} & {sp.latex(sp.Rational(res['spearman_var']).evalf(1))} & "
                        f"{res['quadratic_mean']:.{n3}f} & {sp.latex(sp.Rational(res['quadratic_var']).evalf(1))} \\\\ \\hline\n")
            
            
        f.write(r"\end{tabular}" + "\n")
if __name__ == "__main__":
    os.chdir("отчет")
    normal_sizes = [20, 60, 100]
    normal_rhos = [0, 0.5, 0.9]
    mixture_sizes = [20, 60, 100]

    print("Запуск экспериментов для нормальных распределений...")
    normal_results = run_experiment(generate_normal, normal_sizes, normal_rhos)

    print("\nЗапуск экспериментов для смесевых распределений...")
    mixture_results = run_experiment(generate_mixture, mixture_sizes, [None], is_mixture=True)

    output_img_dir = './img/'

    normal_images = plot_samples_and_ellipse(
        generate_normal, normal_sizes, normal_rhos, output_img_dir
    )

    mixture_images = plot_samples_and_ellipse(
        lambda n, _: generate_mixture(n),
        mixture_sizes,
        ['mixture'],
        output_img_dir
    )

    # save_to_latex(
    #     normal_results, 
    #     './data/normal.tex', 
    #     'normal', 
    #     normal_images,
    # )

    # # Для смеси распределений
    save_to_latex(
        mixture_results, 
        './data/mixture.tex', 
        'mixture', 
        mixture_images
    )
    print("Отчет успешно сгенерирован в папке ./отчет/")