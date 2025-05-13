import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_intervals(x):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return (q1, q3), (x.min(), x.max())

def jaccard_interval(int1, int2):
    low1, high1 = int1
    low2, high2 = int2
    

    inter =  min(high1, high2) - max(low1, low2)

    union =  max(high1, high2) - min(low1, low2)

    return inter / union if union != 0 else 0.0


def sci_notation(x):
    if x == 0:
        return rf"$0$"
    coeff, exp = f"{x:.3e}".split("e")
    if int(exp) == -1:
        return rf"${round(float(coeff)/10, 3)}$"
    if int(exp) == 0:
        return rf"${coeff}$"
    return rf"${coeff}\times10^{{{int(exp)}}}$"


out_dir = "report/results"

n = 1000
np.random.seed(0)
X1 = np.random.normal(loc=0, scale=0.09, size=n)
X2 = np.random.normal(loc=1, scale=0.11, size=n)


plt.figure()
plt.hist(X1, bins=25, alpha = 0.5, label='$X_1$')
plt.hist(X2, bins=25, alpha = 0.5, label='$X_2$')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'hist_plot.png'), dpi=300)
plt.close()

os.makedirs(out_dir, exist_ok=True)

inn1, out1 = compute_intervals(X1)
inn2, out2 = compute_intervals(X2)

df_intervals = pd.DataFrame({
    ' ': ['$X_1$','$X_1$','$X_2$','$X_2$'],
    '   ': ['inn','out','inn','out'],
    'Нижний': [f"{i:.3f}" for i in [inn1[0], out1[0], inn2[0], out2[0]]],
    'Верхний': [f"{i:.3f}" for i in [inn1[1], out1[1], inn2[1], out2[1]]],
})
df_intervals.to_latex(
    os.path.join(out_dir, "intervals.tex"),
    index=False,
    # formatters={"low": sci_notation, "high": sci_notation},
    column_format="llrr",
    caption="Внутренние и внешние интервалы выборок $X_1$ и $X_2$",
    label="tab:intervals",
    escape=False,
    bold_rows=True,
    position="H"
)

a_values = np.linspace(-1.0, 3.0, 21)
j_inn = []
j_out = []
for a in a_values:
    inn1_a = (inn1[0] + a, inn1[1] + a)
    out1_a = (out1[0] + a, out1[1] + a)
    j_inn.append(jaccard_interval(inn1_a, inn2))
    j_out.append(jaccard_interval(out1_a, out2))

df_j = pd.DataFrame({
    'a': [str(round(i, 1)) for i in a_values],
    '$J_{inn}$': list(map(sci_notation, j_inn)),
    '$J_{out}$': list(map(sci_notation, j_out))
})
df_j.to_latex(
    os.path.join(out_dir, "jaccard.tex"),
    index=False,
    # formatters={"low": sci_notation, "high": sci_notation},
    column_format="lrr",
    caption="Значения индексов Жаккара при варьировании параметра $a$",
    label="tab:jaccard",
    escape=False,
    bold_rows=True,
    position="H"
)

plt.figure()
plt.plot(a_values, j_inn, label='$J_{inn}$(a)')
plt.plot(a_values, j_out, label='$J_{out}$(a)')
plt.xlabel('a')
plt.ylabel('Индексы Жаккара')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'jaccard_plot.png'), dpi=300)
plt.close()

a_inn = max(j_inn)
a_out = max(j_out)

index_a_inn = j_inn.index(a_inn)
index_a_out = j_out.index(a_out)

print(f"о_Inn = {a_inn:.4f} (a: {a_values[index_a_inn]}), j_Out = {a_out:.4f} (a: {a_values[index_a_out]})")

import subprocess
os.chdir('report')
subprocess.run(['pdflatex', 'main.tex'], check=True)

