import streamlit as st
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'

st.subheader('確率分布の実験')

"""
### 正規分布

母数（パラメータ）を変化させたときのグラフの変化の確認
"""

# 期待値と分散を指定
mu = st.sidebar.slider('正規分布の期待値', min_value=-5.0, max_value=5.0, step=0.01)
sigma = st.sidebar.slider('正規分布の分散', min_value=0.1, max_value=20.0, step=0.1)

# 標準正規分布の描画
x_1 = np.linespace(-10, 10, 100)
z = stats.norm.pdf(x_1, mu=mu, sigma=sigma)

fig_norm, ax1 = plt.subplots()
ax1.plot(x_1, z, label='std_norm')
ax1.legend()
st.pyplot(fig_norm)

"""
### ポアソン分布

母数（パラメータ）を変化させたときのグラフの変化の確認
"""

# 期待値と分散を指定
lam = st.sidebar.slider('ポアソン分布の期待値', min_value=0, max_value=30, step=1)

# ポアソン分布の描画
x_2 = np.linspace(0, 30, 31)
r = stats.poisson.pmf(x_2, lamda)

fig_pois, ax2 = plt.subplots()
ax2.bar(x_2, height=r, color='#00A968', label='poisson')
ax2.legend()
st.pyplot(fig_pois)

st.subheader('数理モデルの実験')

# SIRモデルの関数
def SIR(t, y, beta, gamma):
    dSdt = -beta * y[0] * y[1]
    dIdt = beta * y[0] * y[1] - gamma * y[1]
    dRdt = gamma * y[1]
    return [dSdt, dIdt, dRdt]

# 初期値の設定
T = 200  # モデルの実行期間
S0 = 999  # 感受性者数_初期値
I0 = 1  # 感染者数_初期値
R0 = 0  # 免疫保持者数_初期値

r0 = st.sidebar.slider('基本再生産数', min_value=1.0, max_value=10.0, step=0.01)
gamma = 1 / st.sidebar.slider('回復率（回復までの日数）', min_value=1, max_value=20, step=1)
beta = r0 * gamma / S0  # 感染率

# 分析
solve = solve_ivp(
    fun=SIR,
    t_span=[0, T],
    y0=[S0, I0, R0],
    args=(beta, gamma),
    dense_output=True,
)

# 分析結果の可視化
plt.figure(figsize=(6, 4))
plt.plot(solve.t, solve.y.T)
plt.xlabel('day')
plt.ylabel('population')
plt.legnd(['Susceptible', 'Infectious', 'Removed'])
st.pyplot(plt)
