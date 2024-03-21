---
title: test
date: 2024-03-20 18:16:14
tags:
---

## 行内公式

- **状态价值函数：**$V_\pi(s_t)~=~\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\Big[Q_\pi(s_t,A_t)\Big]$是衡量使用当前策略$\pi$下，所在(具体观测到的)状态$s_t$的好坏，即所能获得的期望回报。状态价值与动作价值的区别是当前动作$a_t$是否服从我们的策略$\pi$。它即反映状态的好坏，也一定程度上反映我们的策略$\pi$的好坏。

## 行间公式

**引理 7.3. 策略梯度的连加形式** (Page.111)
$$
\begin{aligned}
\frac{\partial J(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}&=\ \mathbb{E}_{S_1,A_1}\left[\boldsymbol{g}(S_1,A_1;\boldsymbol{\theta})\right] \\\\
&+\gamma\cdot\mathbb{E}_{S_1,A_1,S_2,A_2}\left[\boldsymbol{g}(S_2,A_2;\boldsymbol{\theta})\right] \\\\
&+\ldots \\\\
&+\gamma^{n-1}\cdot\mathbb{E}_{S_1,A_1,S_2,A_2,S_3,A_3,\cdotp\cdotp S_n,A_n}\left[\boldsymbol{g}(S_n,A_n;\boldsymbol{\theta})\right]\\\\
&=\mathbb{E}_{S\sim d(\cdot)}\left[\mathbb{E}_{A\sim\pi(\cdot|S;\boldsymbol{\theta})}\left[\sum_{t=1}^n\gamma^{t-1}\cdot \boldsymbol{g}(S_t,A_t;\boldsymbol{\theta})\right]\right].
\end{aligned}
$$

## 图片标题

![小天使](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/03/20/20240320-213333.png)

<center><p class="image-caption">Ijichi Nijika</p></center>
