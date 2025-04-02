import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def least_squares_estimates(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    beta1 = (np.mean(x * y) - mean_x * mean_y) / (np.mean(x**2) - mean_x**2)
    beta0 = mean_y - beta1 * mean_x
    return beta0, beta1

def robust_estimates(x, y):
    n = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    r_Q = np.mean(np.sign(x - med_x) * np.sign(y - med_y))
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    if n % 4 == 0:
        l = n // 4
    else:
        l = n // 4 + 1
    j = n - l + 1
    q_x = x_sorted[j - 1] - x_sorted[l - 1]
    q_y = y_sorted[j - 1] - y_sorted[l - 1]
    beta1_robust = r_Q * (q_y / q_x) if q_x != 0 else 0
    beta0_robust = med_y - beta1_robust * med_x
    return beta0_robust, beta1_robust

def analyze_sample(perturb=False):
    n_points = 20
    a_true = 2
    b_true = 2
    x = np.arange(-1.8, 2.0 + 0.2, 0.2)
    np.random.seed(0)
    noise = np.random.normal(0, 1, size=n_points)
    y = a_true + b_true * x + noise
    if perturb:
        y[0] += 10
        y[-1] -= 10
    a_ls, b_ls = least_squares_estimates(x, y)
    a_rob, b_rob = robust_estimates(x, y)
    return (a_ls, b_ls), (a_rob, b_rob), (a_true, b_true), x, y

(ls_est_nonpert, rob_est_nonpert, true_params, x_nonpert, y_nonpert) = analyze_sample(perturb=False)
(ls_est_pert, rob_est_pert, _, x_pert, y_pert) = analyze_sample(perturb=True)

df_nonpert = pd.DataFrame({
    'Метод': ['МНК', 'МНМ'],
    '$\\hat{a}$': [ls_est_nonpert[0], rob_est_nonpert[0]],
    '$a$': [true_params[0], true_params[0]],
    '$\\hat{b}$': [ls_est_nonpert[1], rob_est_nonpert[1]],
    '$b$': [true_params[1], true_params[1]]
})

df_pert = pd.DataFrame({
    'Метод': ['МНК', 'МНМ'],
    '$\\hat{a}$': [ls_est_pert[0], rob_est_pert[0]],
    '$a$': [true_params[0], true_params[0]],
    '$\\hat{b}$': [ls_est_pert[1], rob_est_pert[1]],
    '$b$': [true_params[1], true_params[1]]
})

output_dir = "отчет/data"
os.makedirs(output_dir, exist_ok=True)

filename_nonpert = os.path.join(output_dir, "df_nonpert_table.tex")
with open(filename_nonpert, 'w', encoding='utf-8') as f:
    f.write(df_nonpert.to_latex(index=False,
                                float_format="%.1f",
                                caption="Результаты для невозмущенной выборки",
                                label="tab:nonpert"))

filename_pert = os.path.join(output_dir, "df_pert_table.tex")
with open(filename_pert, 'w', encoding='utf-8') as f:
    f.write(df_pert.to_latex(index=False,
                             float_format="%.1f",
                             caption="Результаты для возмущенной выборки",
                             label="tab:pert"))

def plot_regression(x, y, ls_params, rob_params, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="black", label="Наблюдения")
    y_ls = ls_params[0] + ls_params[1] * x
    plt.plot(x, y_ls, color="red", label="МНК")
    y_rob = rob_params[0] + rob_params[1] * x
    plt.plot(x, y_rob, color="blue", linestyle="--", label="МНМ")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_regression(x_nonpert, y_nonpert, ls_est_nonpert, rob_est_nonpert,
                "Невозмущенная выборка", "nonpert_plot.png")
plot_regression(x_pert, y_pert, ls_est_pert, rob_est_pert,
                "Возмущенная выборка", "pert_plot.png")

print("Таблица 1: Результаты для невозмущенной выборки")
print(df_nonpert.to_string(index=False))
print("\nТаблица 2: Результаты для возмущенной выборки")
print(df_pert.to_string(index=False))

