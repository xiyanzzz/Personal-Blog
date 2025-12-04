---
title: Diffusion生成模型 (DDPM/Score-based/Flow)
date: 2025-9-04 19:17:44
tags: [Deep Learning, 算法实现, Diffusion]
categories: Generative Models
---
# Diffusion生成模型 (DDPM/Score-based/Flow)

**计划**：

- DPMs
- DDPM
- DDIM
- iDDPM
- SMLD (DENOISING SCORE MATCHING WITH LANGEVIN DYNAMICS) or score-based NCSN
- SDE-based
- Flow Matching
- Rectified Flow

**前置**：

$x_0$始终代表“果”，即从原始分布中抽取到的结果，而一系列潜变量$x_t$或$z$为”因“

后验与先验都是求因的概率，先验常常为已知的高斯分布

后验分布：$p(\theta|x_0)$，知果求因

先验：$p(x_T)$，一般假定已知，例如高斯分布

## 1. 扩散概率模型 (Diffusion Probabilistic Models, DPMs)

Deep Unsupervised Learning using  Nonequilibrium Thermodynamics 2015-11 Jascha Sohl-Dickstein

扩散概率模型（Diffusion Probabilistic Models, DPMs）最早由Sohl‐Dickstein等人在论文“Deep Unsupervised Learning using Nonequilibrium Thermodynamics”中提出，其核心思想是将数据分布通过逐步加入噪声（forward diffusion）推向简单的先验（如高斯白噪声），再学习一个参数化的反向去噪过程（reverse diffusion）来恢复数据分布。

### 1.1. 问题表述

- 假设原始数据: $x_0\in \mathbb R^{d}$，其中$d$为数据维度，如一张$32\times 32$的彩色图片: $d=32\times 32 \times 3$
- 扩散步: 设$T$为扩散总步数(通常几百到一千)，则扩散步取值: $t=1,2,\dots,T$

### 1.2. 正向扩散过程 (Forward Diffusion Process)

正向过程$q$定义了如何在每一步向数据中加入少量高斯噪声，使得在$K$步之后，数据近似服从标准正态分布：

​    In the forward process, noise is incrementally added to the data $x_0$ over $T$ timesteps, creating a sequence of noisy variables $x_1,x_2,\cdots, x_T$. The process is defined as a ==Markov diffusion kernel==:
$$
\begin{equation}
\label{ddpm:fp}
q\left({x}_t \mid {x}_{t-1}\right)=\mathcal{N}\left({x}_t ; \sqrt{1-\beta_t} {x}_{t-1}, \beta_t I\right),
\end{equation}
$$
where $\beta_t$ are variance parameters that control the amount of noise added at each step. 

According to Markov chain, we can get 前向轨迹中变分变量的联合分布:
$$
\begin{equation}
        q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1}).
\end{equation}
$$


通过多步递推，能得到任意时刻$t$与初始$x_0$的封闭形式分布：(参考文献[2]中公式(61-70))
$$
\begin{equation}

q\left({x}_t \mid {x}_{0}\right)=\mathcal{N}\left({x}_t ; \sqrt{\bar{\alpha}_t} {x}_0,\left(1-\bar{\alpha}_t\right) {I}\right),

\end{equation}
$$
where $\alpha_i=(1-\beta_i),~\bar \alpha_t=\prod_{i=1}^t \alpha_i$.  At the end of the forward process, $x_T$ approximates pure Gaussian noise.

> 公式(3)这是定义的前向扩散核的性质，也可看作一种trick。包括后面的DDIM，即使前向不再是markov的，但仍需满足这个性质以供目标函数的推导。

### 1.3. 反向去噪过程 (Reverse Denoising Process)

​    The reverse process aims to reconstruct $x_0$ from $x_T$ by learning a series of denoising steps (近似后验):
$$
\begin{equation}

​    \label{ddpm:rp}

​        p_\theta\left(x_{t-1} \mid x_t\right)=\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right),

​    \end{equation}
$$
​    where $\mu_\theta$ and $\Sigma_\theta$ are parameterized by a neural network.

> "Given continuous diffusion (small step size $\beta$), if $q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)$ is a Gaussian distribution, then $p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)$ will also be a Gaussian distribution. Therefore, the goal of training is to estimate the mean and variance of $p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)$."

​    The model iteratively predicts $x_{t-1}$ from $x_t$ until it reconstructs $x_0$. The joint probability distribution $p_\theta(x_{0:T})$ can be expressed as:
$$
\begin{equation}
\label{eq:dpm1}

​        p_\theta\left(x_{0: T}\right)=p\left(x_T\right) \prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right) .

​    \end{equation}
$$
The reverse process is defined as a Markov chain (Formulate $\eqref{eq:dpm1}$ employs Markov property).

### 1.4. 训练目标

#### 1.4.1. 最大似然

最大化对数似然:
$$
\mathbb{E}_{q(x_0)}\left[ \log p_\theta(x_0)\right]
$$


其中边缘概率密度$p_\theta(x_0)$表示为联合概率密度对潜变量$x_{1:T}$的积分: $p_\theta(x_0)=\int p_\theta(x_{0:T})\mathrm{d}x_{1:T}$，但这些潜变量是"intractable"的，导致积分不能直接计算。于是，引入变分推理。

#### 1.4.2. 变分推理

引入变分分布$q(x_{1:T}|x_0)$，以近似真实的后验分布$p_\theta(x_{1:T}|x_0)$。于是，对数概率密度可写为：
$$
\begin{equation}
        \log p_\theta(x_0)=\log \int q\left(x_{1: T} \mid x_0\right)\frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}  d x_{1: T}.
    \end{equation}
$$
进一步，写成期望的形式:
$$
\begin{equation}
        \log p_\theta\left(x_0\right)=\log \mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right].
    \end{equation}
$$

#### 1.4.3. 推导下界 (ELBO)

利用Jensen' inequality, 将对数移入至期望内:
$$
\begin{aligned}
\log p_\theta\left(x_0\right) &\geq \mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\log \frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right]\\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right) \prod_{t=1}^{T} p_{x \theta}\left(x_{t-1} \mid x_{t}\right)}{\prod_{t=1}^{T} q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right) p_{x \theta}\left(x_{0} \mid x_{1}\right) \prod_{t=2}^{T} p_{x \theta}\left(x_{t-1} \mid x_{t}\right)}{q\left(x_{T} \mid x_{T-1}\right) \prod_{t=1}^{T-1} q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right) p_{x \theta}\left(x_{0} \mid x_{1}\right) \prod_{t=1}^{T-1} p_{x \theta}\left(x_{t} \mid x_{t+1}\right)}{q\left(x_{T} \mid x_{T-1}\right) \prod_{t=1}^{T-1} q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right) p_{x \theta}\left(x_{0} \mid x_{1}\right)}{q\left(x_{T} \mid x_{T-1}\right)}\right]+\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \prod_{t=1}^{T-1} \frac{p_{x \theta}\left(x_{t} \mid x_{t+1}\right)}{q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln p_{x \theta}\left(x_{0} \mid x_{1}\right)\right]+\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right)}{q\left(x_{T} \mid x_{T-1}\right)}\right]+\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\sum_{t=1}^{T-1} \ln \frac{p_{x \theta}\left(x_{t} \mid x_{t+1}\right)}{q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln p_{x \theta}\left(x_{0} \mid x_{1}\right)\right]+\mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right)}{q\left(x_{T} \mid x_{T-1}\right)}\right]+\sum_{t=1}^{T-1} \mathbb{E}_{q\left(x_{1: T} \mid x_{0}\right)}\left[\ln \frac{p_{x \theta}\left(x_{t} \mid x_{t+1}\right)}{q\left(x_{t} \mid x_{t-1}\right)}\right] \\
&=\mathbb{E}_{q\left(x_{1} \mid x_{0}\right)}\left[\ln p_{x \theta}\left(x_{0} \mid x_{1}\right)\right]+\mathbb{E}_{q\left(x_{T-1}, x_{T} \mid x_{0}\right)}\left[\ln \frac{p\left(x_{T}\right)}{q\left(x_{T} \mid x_{T-1}\right)}\right] \\
&=\underbrace{\mathbb{E}_{q\left(x_{1} \mid x_{0}\right)}\left[\ln p_{\theta}\left(x_{0} \mid x_{1}\right)\right]}_{\text {reconstruction term }}-\underbrace{\mathbb{E}_{q\left(x_{T-1} \mid x_{0}\right)}\left[D_{\text {KL }}\left(q\left(x_{T} \mid x_{T-1}\right) \| p\left(x_{T}\right)\right)\right]}_{\text {prior matching term }} \\
&-\sum_{t=1}^{T-1} \underbrace{\mathbb{E}_{q\left(x_{t-1}, x_{t+1} \mid x_{0}\right)}\left[D_{\text {KL }}\left(q\left(x_{t} \mid x_{t-1}\right) \| p_{\theta}\left(x_{t} \mid x_{t+1}\right)\right)\right]}_{\text {consistency term }}

\end{aligned}
$$

\\\待续...推导与下文DDPM重合


## 去噪概率扩散模型 (Denoising Diffusion Probablilistic Models, DDPMs)

Denoising Diffusion Probabilistic Models 2020-10 Jonathan Ho

> DDPM对DPM的改动：不再是预测原始的$x_0$，而是预测每一个时刻添加的噪声，降低了模型的学习难度。
>
> 在原始的DPM模型中，训练的模型就是直接预测$x_0$进而得到$q\left(x_{t-1} \mid x_t, x_0\right)$近似表示。 在后来的 DDPM 改进中，利用了$x_0,x_t,\epsilon_t$ 三者的关系式（公式$\eqref{eq:ddpm5}$）， 用$x_t,\epsilon_t$替换了$x_0$，进而令模型去预测$\epsilon_t$。



DDPMs model the data distribution $p(x_0)$ by defining a forward diffusion process that corrupts data into noise and a reverse denoising process that reconstructs the original data from the noise.

### Forward Diffusion Process

In the forward process, noise is incrementally added to the data $x_0$ over $T$ timesteps, creating a sequence of noisy variables $x_1,x_2,\cdots, x_T$. The process is defined as a ==Markov diffusion kernel==:
$$
\begin{equation}
    \label{ddpm:fp10}
        q\left({x}_t \mid {x}_{t-1}\right)=\mathcal{N}\left({x}_t ; \sqrt{1-\beta_t} {x}_{t-1}, \beta_t I\right),
\end{equation}
$$
where $\beta_t$ are variance parameters that control the amount of noise added at each step.

According to Markov chain, we can get:
$$
\begin{equation}

​    \label{eq:ddpm1}

​        q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1}).

​    \end{equation}
$$
Additionally, the cumulative noise distribution can be expressed as:
$$
\begin{equation}
\label{eq:ddpm12}

​        q\left({x}_t \mid {x}_{0}\right)=\mathcal{N}\left({x}_t ; \sqrt{\bar{\alpha}_t} {x}_0,\left(1-\bar{\alpha}_t\right) {I}\right),

​    \end{equation}
$$
​    where $\bar \alpha_t=\prod_{i=1}^t (1-\beta_i)$.  At the end of the forward process, $x_T$ approximates pure Gaussian noise.

### Reverse Denoising Process

The reverse process aims to reconstruct $x_0$ from $x_T$ by learning a series of denoising steps:
$$
\begin{equation}

​    \label{eq13}

​        p_\theta\left(x_{t-1} \mid x_t\right)=\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right),

​    \end{equation}
$$


​    where $\mu_\theta$ and $\Sigma_\theta$ are parameterized by a neural network.

​    The model iteratively predicts $x_{t-1}$ from $x_t$ until it reconstructs $x_0$. The joint probability distribution $p_\theta(x_{0:T})$ can be expressed as:
$$
\begin{equation}

​    \label{eq:ddpm2}

​        p_\theta\left(x_{0: T}\right)=p\left(x_T\right) \prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right) .

​    \end{equation}
$$
​    Here,

- $p(x_T)$ is the ==prior distribution== of $x_T$, usually a standard Gaussian.
- $p_\theta\left(x_{t-1} \mid x_t\right)$ is the reverse process, parameterized by $\theta$.
- The reverse process is defined as a Markov chain (Formula $\eqref{eq:ddpm2}$ employs Markov property).

### Derivation of the DDPM optimization objective

1. **The Problem Setup:** Marginal Likelihood of the Data

The objective is to maximize the probability of the potential $x_0$ from original data distribution under the generative model:
$$
 \begin{equation}

​        \mathbb{E}_{q(x_0)}\left[ \log p_\theta(x_0)\right],

​    \end{equation}
$$
where $p_\theta(x_0)=\int p_\theta(x_{0:T})\mathrm{d}x_{1:T}$, and $x_{0:T}$ represents the sequence of variables $\{x_0,x_1,\cdots,x_T\}$.

​    Since the marginal likelihood involves integration over ==latent variables== $x_{1:T}$, this integral is generally intractable to compute directly.

2. **Variational Inference:** Introducing $q(x_{1:T}|x_0)$

To address this intractability, we introduce a ==variational distribution== $q(x_{1:T}|x_0)$, which acts as an approximation to the true ==posterior== $p_\theta(x_{1:T}|x_0)$. Using this, we can re-express the log-likelihood as :
$$
\begin{equation}

​        \log p_\theta(x_0)=\log \int \frac{q\left(x_{1: T} \mid x_0\right)}{q\left(x_{1: T} \mid x_0\right)} p_\theta\left(x_{0: T}\right) d x_{1: T}.

​    \end{equation}
$$
​    This allows us to rewrite the integral as an expectation under $q\left(x_{1: T} \mid x_0\right)$:
$$
\begin{equation}

​        \log p_\theta\left(x_0\right)=\log \mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right].

​    \end{equation}
$$

3. **Jensen's Inequality:** Deriving the ELBO

Directly computing this expectation is still intractable because the logarithm is outside the expectation. To simplify, we move the logarithm inside the expectation by applying Jensen's inequality ==\cite{bishop2006pattern}==:
$$
  \begin{equation}

​        \log p_\theta\left(x_0\right) \geq \mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\log \frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right].

​    \end{equation}
$$


​    Therefore, we can get the evidence lower bound (ELBO) of the model log-likelihood:
$$
    \begin{equation}

​    \label{eq:ddpm3}

​        \mathbb{E}_{q(x_0)}\left[ \log p_\theta(x_0)\right]\geq \mathbb{E}_{q\left(x_{0: T}\right)}\left[\log \frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right]=:\mathcal{L}_{\text {ELBO }}.

​    \end{equation}
$$

4. **Expanding the ELBO**

 Substitute the expressions $\eqref{eq:ddpm1}$ and $\eqref{eq:ddpm2}$ into the ELBO $\eqref{eq:ddpm3}$ and breaking it into terms:

$$
\begin{equation}

​        \mathcal{L}_{\mathrm{ELBO}}=\mathbb{E}_{q\left(x_{0: T}\right)}\left[\log p\left(x_T\right)+\sum_{t=1}^T \log \frac{p_\theta\left(x_{t-1} \mid x_t\right)}{q\left(x_t \mid x_{t-1}\right)} \right].

​    \end{equation}
$$


​    Employing Markov property: $q\left(x_t \mid x_{t-1}\right)=q\left(x_t \mid x_{t-1},x_0\right)$ and ==Bayes's formulation==: $q\left(x_t \mid x_{t-1},x_0\right)=\frac{q\left(x_{t-1} \mid x_{t},x_0\right) q\left(x_t \mid x_0\right)}{q\left(x_{t-1} \mid x_0\right)}$, and extracting the $t=1$ term from the summation, one obtains:

$$
    \begin{equation}

​        \mathcal{L}_{\mathrm{ELBO}}=\mathbb{E}_{q\left(x_{0: T}\right)} \left[ \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t=2}^T \log \frac{p_\theta\left(x_{t-1} \mid x_t\right)}{q\left(x_{t-1} \mid x_{t},x_0\right)} + \log p_\theta(x_0|x_1) \right].

​    \end{equation}
$$


​    Furthermore, it can be interpreted as:

$$
   \begin{equation}

​        \begin{split}

​            \mathcal{L}_{\mathrm{ELBO}}&=\mathbb{E}_{q\left(x_{0: T}\right)} \left[- D_{\mathrm{KL}}\left(q\left(x_T \mid x_0\right) \| p\left(x_T\right)\right)- \right.\\&- \sum_{t=2}^T D_{\mathrm{KL}}\left(q\left(x_{t-1} \mid x_t, x_0\right) \| p_\theta\left(x_{t-1} \mid x_t\right)\right)+\\&+ \left.\log p_\theta\left(x_0 \mid x_1\right) \right].

​        \end{split}

​    \end{equation}
$$
### Loss of DDPM

The loss of DDMP to be minimized can be expressed negative ELBO:
$$
    \begin{equation}

​        \begin{split}

​            L_{\text{DDPM}}&=\mathbb{E}_{q\left(x_{0: T}\right)} \left[ D_{\mathrm{KL}}\left(q\left(x_T \mid x_0\right) \| p\left(x_T\right)\right)+ \right.\\&+ \sum_{t=2}^T D_{\mathrm{KL}}\left(q\left(x_{t-1} \mid x_t, x_0\right) \| p_\theta\left(x_{t-1} \mid x_t\right)\right)-\\&- \left.\log p_\theta\left(x_0 \mid x_1\right) \right].

​        \end{split}

​    \end{equation}
$$




​    Each term corresponds to:

1. The first Term: constant term

$$
\begin{equation}
            D_{\mathrm{KL}}\left(q\left(x_T \mid x_0\right) \| p\left(x_T\right)\right),
        \end{equation}
$$



​        This term is constant w.r.t. the learnable parameters $\theta$, so it does not contribute to the optimization. It can be ignored during training.

2. The second term: reverse process term
   $$
   \begin{equation}
   
   ​            D_{\mathrm{KL}}\left(q\left(x_{t-1} \mid x_t, x_0\right) \| p_\theta\left(x_{t-1} \mid x_t\right)\right),
   
   ​        \end{equation}
   $$
   

   ​        encouraging the reverse process $p_\theta\left(x_{t-1} \mid x_t\right)$ to match the true posterior $q\left(x_{t-1} \mid x_t,x_0\right)$.

   ​        Actually, we can simplify this term using the fact that $q\left(x_{t-1} \mid x_t,x_0\right)$ is a ==Gaussian==:
   $$
   \begin{equation}
   
   ​        \label{eq:ddpm4}
   
   ​            q\left(x_{t-1} \mid x_t, x_0\right)=\mathcal{N}\left(x_{t-1} ; \tilde{\mu}_t\left(x_t, x_0\right), \tilde{\beta}_t I\right),
   
   ​        \end{equation}
   $$
   

   ​        where 
   $$
    \begin{equation}
   
   ​            \tilde{\mu}_t\left(x_t, x_0\right)=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0+\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} x_t,
   
   ​        \end{equation}
   $$
   and
   $$
    \begin{equation}
   
   ​            \tilde{\beta}_t=\frac{\beta_t\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t}.
   
   ​        \end{equation}
   $$
   

   ​        Thus, the KL divergence between $p_\theta\left(x_{t-1} \mid x_t\right)$ and $q\left(x_{t-1} \mid x_t,x_0\right)$ can be computed explicitly for each timestep $t$. 

3. The third term: reconstruction term

$$
  \begin{equation}

​            -\log p_\theta\left(x_0 \mid x_1\right).

​        \end{equation}
$$



​        This is the reconstruction loss term from the last denoising step, and it can be ignored during training since it can be approximated using the same neural networks.

4. **Simplified Training Objective**

 Instead of directly parameterizing $\mu_\theta(x_t,t)$, the model predicts the noise $\epsilon$ added in the forward process:
$$
\begin{equation}

​    \label{eq:ddpm5}

​        x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) .

​    \end{equation}
$$


​    The model learns to predict $\epsilon$ by minimizing the following loss:
$$
\begin{equation}

​        {L}_{\text {simplified }}=\mathbb{E}_{t, x_0, \epsilon}\left[\left\|\epsilon-\epsilon_\theta\left(x_t, t\right)\right\|^2\right].

​    \end{equation}
$$


​    This objective is equivalent to the original $L_\text{DDPM}$ but is easier to compute and highly effective in practice.

Combining Formula $\eqref{eq:ddpm4}$ and $\eqref{eq:ddpm5}$, and substituting $\epsilon$ with estimation $\epsilon_\theta$, we get the sampling function:
$$
\begin{equation}

​    x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} {\epsilon}_\theta\left(x_t, t\right)\right)+\sigma_t z,

\end{equation}
$$


where $z\sim \mathcal N(0,I)$ from a reparameterization trick and $\sigma_t$ is a time dependent constants that can be set as $\beta_t$ or $\tilde{\beta}_t$ (the results is experimentally similar).

![image-20251112161839601](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2025/11/12/20251112-161840.png)

## 去噪扩散隐式模型 (Denoising Diffusion Implicit Models,DDIMs)

在DDPM的ELBO展开推导过程中，用到了以下性质：

- 前向过程联合概率$q(x_{1:T}|x_0)$的分解$\eqref{eq:ddpm1}$ 和逆向过程$p_\theta\left(x_{0: T}\right)$的分解 $\eqref{eq:ddpm2}$ 
- 前向过程线性高斯的特性$ q\left({x}_t \mid {x}_{0}\right)$ $ \eqref{eq:ddpm12}$
- 贝叶斯：$q\left(x_{t-1} \mid x_t, x_0\right)$ $\eqref{eq:ddpm4}$

观察DDPM最终ELBO的展开式，最重要的就是让我们的模型$p_\theta\left(x_{t-1}\mid x_{t}\right)$去近似逆扩散的转换核$q\left(x_{t-1} \mid x_t, x_0\right)$，而该逆扩散核是由前向过程马尔科夫形式的分解来获取的，那我们是否可以直接定义这个逆扩散核的形式，反过来推前向过程，使其不必要依赖马尔科夫性质，找到更通用的问题陈述。

为了保持最终的ELBO$\eqref{eq:ddpm3}$推导不变，其前向过程(即分母部分$q(x_{1:T}|x_0)$)最终需满足：
$$
\label{eq:ddim1}
q_{\sigma}\left(x_{1: T} \mid x_{0}\right):=q_{\sigma}\left(x_{T} \mid x_{0}\right) \prod_{t=2}^{T} q_{\sigma}\left(x_{t-1} \mid x_{t}, x_{0}\right)
$$
这在原始定义中用到了前向的马尔科夫性质以及$q\left(x_{t-1} \mid x_t, x_0\right)$ $\eqref{eq:ddpm4}$的贝叶斯展开。实际上对DDPM的前向分解再变换一下可直接得到我们想要的通用分解形式：
$$
\begin{equation}
\begin{aligned}
q(x_{1:T}|x_0)&=\prod_{t=1}^Tq(x_t|x_{t-1})\\
&=q(x_1\mid x_0)\prod_{t=2}^Tq(x_t|x_{t-1},x_0)\\
&=q(x_1\mid x_0)\prod_{t=2}^T \frac{q(x_{t-1}|x_{t},x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}\\
&=q\left(x_{T} \mid x_{0}\right) \prod_{t=2}^{T} q\left(x_{t-1} \mid x_{t}, x_{0}\right)
\end{aligned}
\end{equation}
$$


观察式$\eqref{eq:ddim1}$，前向过程的分解直接用到了逆扩散核和后验分布，后验分布不妨与DDPM保持一致，即：
$$
q_{\sigma}\left(x_{T} \mid x_{0}\right) \sim \mathcal{N}\left(\sqrt{\bar{\alpha}_{T}} x_{0},\left(1-\bar{\alpha}_{T}\right) I\right)
$$
接着直接定义逆扩散核的形式：
$$
q_{\sigma}\left(x_{t-1} \mid x_{t}, x_{0}\right) \sim \mathcal{N}(\underbrace{\sqrt{\bar{\alpha}_{t-1}} x_{0}+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_{t}^{2}} \cdot \frac{x_{t}-\sqrt{\bar{\alpha}_{t}} x_{0}}{\sqrt{1-\bar{\alpha}_{t}}}}_{\text {期望 }}, \underbrace{\sigma_{t}^{2} I}_{\text {方差 }})
$$
这时我们巧妙的将逆扩散核的方差设置成了自由参数，当$\sigma_t=0$，逆过程便不再具有随机性，无需一步步采样，这便是DDIM加速采样的核心。

注意，该形式并不是随便设定的，为了满足训练过程一致，从$x_0$到任意$x_t$的一步采样性质仍需保留，而以上定义可证明$q\left({x}_t \mid {x}_{0}\right)$ $\eqref{eq:ddpm12}$保持不变，这一性质的保留让DDIM和DDPM一样可以直接预测噪声$\epsilon_t$而非$x_0$。



## 噪声条件分数网络 (Noise Conditional Score Networks, NCSNs)

Generative Modeling by Estimating Gradients of the Data Distribution, Yang Song, 2020.10

**引言：**生成模型应用广泛，但现有似然基 (likelihood-based) 方法和 GAN 存在局限性，提出基于数据分布梯度（分数）的生成模型框架，通过分数匹配估计分数、朗之万动力学采样，并用噪声扰动和退火策略解决关键挑战。

**定义：**对于样本$x\in \mathbb R^{d}$（例如，图片$\mathbb R^{H\times W\times C}$），$P_{data}(x)$表示所有像素值的组合在自然界（训练集）中的联合概率密度，一般为未知分布。独立同分布的观测样本集：$\{\mathbf x_i\}^N_{i=1}$

避免直接学习复杂的分布$P_{data}(x)$，而是直接学习其分数（log 概率密度的梯度）：
$$
s(x)=\nabla_{x} \log p_{data}(x), \quad x \in \mathbb{R}^{d}，
$$
它指明了如何“沿着概率密度上升”的方向移动。

### 分数匹配以估计分数：降噪分数匹配

模型用一个神经网络$s_\theta(x)$来近似这个分数，目标函数表示为：
$$
\label{eq:nscn1}
J(\theta)=\frac{1}{2} \mathbb{E}_{p_{\text {data }}}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})\right\|_{2}^{2}\right]
$$

> 如果两个分布的对数密度梯度是一样的，那么他们的概率密度也是一样的。

但直接计算 $\nabla_x\log p_\theta(x)$需要$p_\theta(x)$的归一化常数。可证明，最小化以上目标函数可以等效于最小化(Hyvärinen, 2005)：
$$
\mathbb{E}_{p_{\text {data }}}\left[\text{tr}\left (\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right) + \frac{1}{2} \left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right\|_{2}^{2}\right]
$$
其中$\nabla_x s_\theta(x)\in \mathbb R^{d \times d}$是$\log p_\theta(x)$的二阶偏导，也称为Hessian 矩阵或$s_\theta(x)$的Jacobian 矩阵，实际计算中开销大。

为了解决批量计算分数匹配开销问题，Vincent (2011) 提出 **Denoising Score Matching**：

首先，对观测样本$x$施加预定义的扰动核$q_{\sigma}(\tilde{x} \mid x)$，获得噪声数据(perturbed data)$\tilde{x}$, 其边缘分布表示为
$$
q_{\sigma}(\tilde{x}) \triangleq \int q_{\sigma}(\tilde{x} \mid x) p_{\mathrm{data}}(x) d x
$$
定义扰动核$q_{\sigma}(\tilde{x} \mid x)=\mathcal N(x,\sigma^2 I)$，由重参数化技巧可得：

$$
\tilde x=x+\sigma \epsilon,~\epsilon\sim\mathcal N(0,I)
$$

利用分数匹配算法获取噪声数据分数的近似：
$$
\begin{aligned}
J(\theta)&= \frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x)}\left[\left\|s_{\theta}(\tilde{x})-\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x})\right\|_{2}^{2}\right] \\
&=  \frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x) p_{\text {data }}(x)}\left[\left\|s_{\theta}(\tilde{x})-\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x)\right\|_{2}^{2}\right]

\end{aligned}
$$
当扰动核定义为上述正态分布时，一阶偏导项$\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x)$可解析地写出来：$\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x)=-\frac{1}{\sigma^2}(\tilde x-x)$

当且仅当扰动强度$\sigma$很小，扰动数据分布$q_\sigma(\tilde x)\approx p_{\text{data}}(x)$，扰动数据分数网络的估计值$s_{\theta}(\tilde{x})$可近似原始数据$p_{\text{data}}(x)$的分数。添加少量噪声无意义，因为我们还是无法得知准确分布$q_\sigma(\tilde x)$，但若我们连续添加噪声，目标就变为学习一条从简单噪声分布$p_T(x)$（如标准正态 $\mathcal{N}(0,I)$到数据分布$p_{\rm data}(x)$的演化路径$\{p_t(x)\}_{t\in[0,T]}$。



### 基于分数的采样：郎之万动力学

**Langevin SDE** 给出如下连续时间随机微分方程：
$$
\mathrm{d} x_{t}=\frac{1}{2} \nabla_{x} \log p\left(x_{t}\right) \mathrm{d} t+\mathrm{d} w_{t}, \quad x_{t} \in \mathbb{R}^{d}(d,),
$$
其中$w_{t}$是标准$d-$维布朗运动（**Brownian motion**）。当系统达到平衡时，轨迹分布收敛到未知目标$p(x)$，这使得我们可以从分数方程中采样，而不依靠其概率密度函数。

为了在实践中数值求解，上述 SDE 通常使用欧拉–马尔可夫（Euler–Maruyama）方法离散化：

1. 选择步长$\eta>0$。
2. 从初始点$x^{(0)}$（可为随机噪声）开始，迭代

$$
x^{(k+1)}=x^{(k)}+\frac{\eta}{2} \nabla_{x} \log p\left(x^{(k)}\right)+\sqrt{\eta} z^{(k)}, \quad z^{(k)} \sim \mathcal{N}\left(0, I_{d}\right)
$$

其中$x^{(k)}\in\mathbb{R}^d$。

**特点**：

- 简单易行，不需要接受拒绝(Metropolis-Hastings accept-reject)步骤（“未校正”）；
- 若步长过大，会引入偏差；若过小，则收敛缓慢。

### 基于分数的生成模型面对的挑战

1. **流形假说**（Manifold Hypothesis）(对于训练)

**通俗比喻**
 想象你有一大堆高质量的手写数字图片，都是28×28像素的灰度图，每个像素有一个亮度值。虽然每张图都在$\mathbb R^{28\times28}$这个「625 维空间」里，但实际上真正有意义的手写数字，只会落在某个「皱皱巴巴的薄薄纸片」上——那是一条远比 625 维小得多的低维曲面（manifold）。你可以把这条曲面想象成「只包含合理笔画组合」的区域，曲面以外几乎没有真实数字存在。

**为什么是挑战？**

- **分数（score）只在数据附近有意义**：我们想学的是数据分布的梯度$\nabla_x\log p(x)$，但这个梯度在高维空间的大多数位置（也就是曲面以外）根本没定义或没样本。网络只能靠曲面附近的点去“猜”，离得太远就瞎猜。
- **学习难度**：如果模型不知道哪条“皱纸片”才是真正的数据所在，就很难在无噪声采样时“一路爬”回到那条皱纸片上，可能就会跑到完全不合理的区域。

---

2. **低密度区域**（Low Data Density Regions）(对于采样)

**通俗比喻**
 再想象一张世界地图，上面标了所有大城市的位置——这些点代表高密度区域。城市之间的空旷地区（大草原、沙漠）就是低密度区域。我们想让模型学会“从空白地图”走到一个城市，但在城市之间漫长的无人的地带，它很难判断方向。

**为什么是挑战？**

- **样本少，信号弱**：在高密度区域（城市），数据多，模型能学到稳定的梯度信号；到了低密度区域（荒原），几乎没有样本，梯度估计会很噪声、很不可靠。
- **采样不稳定**：Score-based 方法里会在各个噪声强度下做梯度上升＋小随机扰动。当走到低密度区域，模型根本不知道往哪里“爬”，就容易乱晃，或者直接飘到乱七八糟的地方。



### Noise Conditional Score Networks

**解决思路（Briefly）**

- **对流形做“平滑扩展”**：用扩散过程（diffusion）把原始流形往周围「吹散」一点，让模型在稍微离开曲面的位置也能看到样本，这样梯度不会一出流形就「断电」。
- **分层噪声 / Annealed Langevin**：先在高噪声下让分布变得“扁平”，样本更遍布空间；然后逐步降低噪声、收拢到曲面，从而避免一开始就深陷低密度荒原。

**NCSNs思想：**添加多个不同水平的噪声，获得一系列噪声数据的中间分布，然后训练一个噪声水平条件化的分数网络，来同时估计不同噪声强度分布的分数。然后用退火郎之万动力学不断从高强度到低强度采样噪声数据，直到可以近似原数据分布。

---

**训练：**

选定一组长度为L的噪声强度数列：$\{\sigma_i\}_{i=1}^{L}$，其中$\frac{\sigma_{i-1}}{\sigma_{i}}>1$，$\sigma_1$最强。
$$
q_{\sigma_i}(\tilde{x} \mid x)=\mathcal N(x,\sigma_i^2 I),~\tilde x=x+\sigma_i \epsilon,~\epsilon\sim\mathcal N(0,I)
$$
损失：
$$
\mathcal{L}\left(\theta ;\left\{\sigma_{i}\right\}_{i=1}^{L}\right) \triangleq \frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \ell\left(\theta ; \sigma_{i}\right)
$$
其中，
$$
\begin{aligned}
\ell\left(\theta ; \sigma_{i}\right) & \triangleq \frac{1}{2} \mathbb{E}_{p_{\text {data }}(x)} \mathbb{E}_{\tilde{x} \sim \mathcal N(x, \sigma_{i}^{2} I)}\left[\left\|s_{\theta}\left(\tilde{x}, \sigma_{i}\right)-\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x)\right\|_{2}^{2}\right] \\
& =\frac{1}{2} \mathbb{E}_{p_{\text {data }}(x)} \mathbb{E}_{\tilde{x} \sim \mathcal N(x, \sigma_{i}^{2} I)}\left[\left\|s_{\theta}\left(\tilde{x}, \sigma_{i}\right)+\frac{\tilde{x}-x}{\sigma_{i}^{2}}\right\|_{2}^{2}\right]
\end{aligned}
$$
权重$\lambda(\sigma_i)=\sigma_i^2$，确保$\ell$与$\sigma_i$无关，使不同噪声的损失维持相同量纲（$\sigma_i\left\|s_{\theta}\left(\tilde{x}, \sigma_{i}\right)\right\|_2\propto 1$ and $\frac{\tilde{x}-x}{\sigma_{i}}\propto \mathcal N(0,I)$）。

---

**采样：**

Annealed Langevin Dynamics

1. 初始化：$x^{(0)}\sim \mathcal N(0,\sigma_1^2I)$

2. for i = 1-L（高噪声到低噪声）:

   - for t = 1-T （对各个水平执行T步郎之万采样）

     - $x^{(t)}=x^{(t-1)}+\frac{\eta_i}{2}s_\theta(x^{(t-1)},\sigma_i)+\sqrt{\eta_i}z^{(t-1)}$

     - where $\eta_i=\eta\cdot \frac{\sigma_i^2}{\sigma_L^2}$为步长，由大到小

   - $x^{(0)}\leftarrow x^{(T)}$: 将这一级的最终采样做下一级的初始采样

3. 当$\sigma_L\approx 0$时，L*T步迭代后的采样结果近似采样于原数据分布$p_{\text{data}}(x)$。

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2025/11/14/20251114-155322.png" alt="image-20251114155321562" style="zoom:50%;" />

---

**改进：**

Improved Techniques for Training Score-Based Generative Models, 2020-10, song

针对高分辨率图像，同作者随机发表了新的论文，提出5点改进



## 基于SDE的分数基方法

SCORE-BASED GENERATIVE MODELING THROUGH  STOCHASTIC DIFFERENTIAL EQUATIONS, 2021-02, Song Yang

>  We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise.

### 基于分数模型的SDE统一框架

1. **前向 SDE（Forward SDE）**：定义一个从数据分布到标准噪声的连续时间随机微分方程$\{x(t)\}_{t=0}^{T}$：

$$
\mathrm{d} x=f(x, t) \mathrm{d} t+g(t) \mathrm{d} w_{t}, \quad x(0) \sim p_{\text {data. }}
$$

其中，$f(x,t)\in \mathbb R^{d}$为漂移(drift)系数，$g(t)\in \mathbb R$为扩散(diffusion)系数，$w_t$表示标准温拿过程(Winer process)。

从时刻$s$到时刻$t$，变量转换的条件概率表示为$p_{st}(x(t)\mid x(s))$， 也可以称为$x(s)$到$x(t)$的转换核（transition kernel）。

2. **逆向SDE（Reverse SDE）**：利用 Girsanov 定理，给出精确对应的逆扩散过程:

$$
\mathrm{d} x=\left[f(x, t)-g(t)^{2} \nabla_{x} \log p_{t}(x)\right] \mathrm{d} t+g(t) \mathrm{d} \bar{w}_{t} .
$$

其中$\bar{w}_{t}$是从$T\to 0$的逆时间布朗运动，$\mathrm d t$符号为负，$p_{t}(x)$等价于$p(x(t))$。只要任意时刻$\nabla_{x} \log p_{t}(x)$已知，便能利用Reverse SDE采样。

3. **分数估计SDE**：连续时间的去噪分数匹配：

$$
\mathbb E_t\left\{ \lambda(t)\mathbb E_{p_0(x)}\mathbb E_{p_{0t}(x(t)\mid x(0))}\left[ \| s_\theta(x(t),t)-\nabla_{x(t)} p_{0t}(x(t)\mid x(0))\|_2^2 \right] \right\}
$$

其中$s_\theta(x(t),t)是$time-dependent score-based model

> 基于以上框架，可以说所有的离散 Diffusion Model（如 DDPM）和 Noise-Conditional Score Network（NCSN）都能看作对某个 SDE 在不同离散化方案下的实例化。

### VE-SDE: SMLD(NCSN)的极限连续

1. 离散扩散核

在SMLD原本的离散前向扩散中，预先定义一系列单调增的噪声标准差$\{\sigma_i\}_{i=0}^T$，转换核表示为：
$$
q_{\sigma_i}({x}_i \mid x_0)=\mathcal N(x_0,\sigma_i^2 I)~ \Rightarrow x_i=x_0+\sigma_i \epsilon,~\epsilon\sim\mathcal N(0,I)
$$
容易证明相邻两个噪声级别间关系为**加性高斯扰动**：
$$
q\left(x_{i} \mid x_{i-1}\right)=\mathcal{N}\left(x_{i} ; x_{i-1}, \sigma_{i}^{2}-\sigma_{i-1}^{2} I\right), \quad i=1, \ldots, T,
$$
其中$\sigma_0=0$, $x_0\sim p_{\text{data}}(x)$。等价地：
$$
x_{i}=x_{i-1}+\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}} z_{i}, \quad z_{i} \sim \mathcal{N}(0, I)
$$


2. 极限连续

令时间$t\in [0,1]$，并把$\sigma_i$看作$\sigma(t_i)$，其中$t_i=i/T$。当$T\to \infin$且$\Delta t=1/T\to0$，有
$$
\sigma_{i}^{2}-\sigma_{i-1}^{2}=\underbrace{\sigma^{2}\left(t_{i}\right)-\sigma^{2}\left(t_{i}-\Delta t\right)}_{\Delta \sigma\to \mathrm d\sigma} \approx \frac{\mathrm{d}}{\mathrm{~d} t} \sigma^{2}(t) \Delta t
$$
于是将离散更新
$$
x(t)-x(t-\Delta t)=\sqrt{\frac{\mathrm{d}}{\mathrm{~d} t} \sigma^{2}(t) \Delta t}\cdot z(t)
$$
视作 SDE 的欧拉–马尔可夫（Euler–Maruyama）离散化，即得
$$
\mathrm{d} x=\underbrace{0}_{f_{\mathrm{VE}}(x, t)} \mathrm{d} t+\underbrace{\sqrt{\frac{\mathrm{d}}{\mathrm{~d} t} \sigma^{2}(t)}}_{g_{\mathrm{VE}}(t)} \mathrm{d} w_{t} .
$$
这就是 **Variance‐Exploding (VE) SDE**

### VP-SDE: DDPM的连续极限

#### DDPM基于分数的解释

DDPM 的最常用训练损失是：
$$
{L}=\mathbb{E}_{t, x_0, \epsilon}\left[\left\|\epsilon-\epsilon_\theta\left(x_t, t\right)\right\|^2\right].
$$
其中，前向过程决定了如下从原始数据到噪声数据的转换核：
$$
{x}_t =\sqrt{\bar{\alpha}_t} {x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon
$$
这是一个去噪 AutoEncoder 风格的目标：预测加到样本上的噪声。

由正态分布表达式，容易证明：
$$
\nabla_{x_{t}} \log q\left(x_{t} \mid x_{0}\right)=-\frac{1}{\sqrt{1-\bar{\alpha}_{t}}} \epsilon
$$
于是可得：
$$
s_{t}\left(x_{t}\right)=\nabla_{x_{t}} \log p_{t}\left(x_{t}\right) \approx-\frac{1}{\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{\theta}\left(x_{t}, t\right)
$$

> DDPM 中的噪声预测网络$\epsilon_\theta$实际上是在学习 score（分数）

1. 离散扩散核

DDPM 采用的正向扩散是一个收缩‐加噪过程，每步定义为
$$
q\left(x_{i} \mid x_{i-1}\right)=\mathcal{N}\left(x_{i} ; \sqrt{1-\beta_{i}} x_{i-1}, \beta_{i} I\right), \quad i=1, \ldots, T .
$$
等价更新写作
$$
x_{i}=\sqrt{1-\beta_{i}} x_{i-1}+\sqrt{\beta_{i}} z_{i}, \quad z_{i} \sim \mathcal{N}(0, I) .
$$

2. 连续极限

令时间$t\in [0,1]$，**$\beta_i=\bar\beta(t_i)\cdot \Delta t$**，其中$t_i=i/T$。当$T\to \infin$且$\Delta t=1/T$。

当$\Delta t\to0$，有：

- $\sqrt{1-\beta_i}=\sqrt{1-\bar\beta(t_i)\Delta t} \approx 1-\frac{1}{2}\bar\beta(t_i)\Delta t$ (泰勒展开：$\sqrt{1-x}\approx 1-\frac{1}{2}x-\frac{1}{8}x^2-\cdots$)
- $\sqrt{\beta_i}=\sqrt{\bar \beta(t_i)\Delta t}=\sqrt{\bar \beta(t_i)}\sqrt{\Delta t}$

离散更新可写为
$$
x(t)-x(t-\Delta t)=-\frac{1}{2}\bar \beta(t)\Delta t~x(t)+\sqrt{\bar\beta(t)}\sqrt{\Delta t}~z(t)
$$
对应的 SDE（Euler–Maruyama 极限）就是（原本的$\bar\beta(t)$写作$\beta(t)$）：
$$
\mathrm{d} x=\underbrace{-\frac{1}{2} \beta(t) x}_{f_{\mathrm{VP}}(x, t)} \mathrm{d} t+\underbrace{\sqrt{\beta(t)}}_{g_{\mathrm{VP}}(t)} \mathrm{d} w_{t} .
$$
这就是 **Variance‐Preserving SDE**

### sub-VP SDE

（通过常微分方程的解）可证明VE SDE的方差会随时间$t\to \infin$而爆炸，而VP SDE的方差总是有界的（当初始分布的方差有界）。这就是Variance Exploding (VE) SDE and Variance Preserving (VP) SDE命名的依据。

随后，作者通过修改VP SDE的方差部分（期望保持一致）提出另一种扩散SDE：sub-VP SDE：
$$
\mathrm{d} {x}=-\frac{1}{2} \beta(t) {x} \mathrm{d} t+\sqrt{\beta(t)\left(1-e^{-2 \int_{0}^{t} \beta(s) \mathrm{d} s}\right)} \mathrm{d} {w} .
$$
其方差始终以VP SDE为上界，且一样随着$t\to\infin$而趋于单位方差。

三种SDE的扩散核可表示为正态分布：
$$
p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0))=\left\{\begin{array}{ll}
\mathcal{N}\left(\mathbf{x}(t) ; \mathbf{x}(0),\left[\sigma^{2}(t)-\sigma^{2}(0)\right] \mathbf{I}\right), & (\mathrm{VE} S D E) \\
\mathcal{N}\left(\mathbf{x}(t) ; \mathbf{x}(0) e^{-\frac{1}{2} \int_{0}^{t} \beta(s) \mathrm{d} s}, \mathbf{I}-\mathbf{I} e^{-\int_{0}^{t} \beta(s) \mathrm{d} s}\right) & (\mathrm{VP} S D E) \\
\mathcal{N}\left(\mathbf{x}(t) ; \mathbf{x}(0) e^{-\frac{1}{2} \int_{0}^{t} \beta(s) \mathrm{d} s},\left[1-e^{-\int_{0}^{t} \beta(s) \mathrm{d} s}\right]^{2} \mathbf{I}\right) & (\text { sub-VP SDE })
\end{array}\right.
$$

### 采样：求解逆时间SDE

#### 通用的SDE数值求解器

对于给定的前向SDE
$$
\mathrm{d} x=f(x, t) \mathrm{d} t+g(t) \mathrm{d} w_{t}, \quad x(0) \sim p_{\text {data }}
$$
通过 Girsanov 变换可得对应的逆向过程
$$
\mathrm{d} x=\left[f(x, t)-g(t)^{2} \nabla_{x} \log p_{t}(x)\right] \mathrm{d} t+g(t) \mathrm{d} \bar{w}_{t} .
$$
其中，$\nabla_{x} \log p_{t}(x)\approx s_\theta(x,t)$由训练好的 **Score Network** 给出。

- *Ancestral sampling*： 原始的DDPM采样方法，针对特殊的VP SDE，对不同SDE需要重新推导
- ***reverse diffusion samplers***： 对以下形式的离散前向过程$x_i=x_{i-1}+f(x_{i-1},t_{i-1})+g(t_{i-1})z_{i-1}$，其逆时间SDE的离散表示为：

$$
x_{i-1}=x_{i}+\left[f\left(x_{i}, t_{i}\right)-\underbrace{g\left(t_{i}\right)^{2}}_{gg^T} s_{\theta}\left(x_{i}, t_{i}\right)\right] \Delta t_{i}+g\left(t_{i}\right) \sqrt{\Delta t_{i}} z_{i}, \quad z_{i} \sim \mathcal{N}\left(0, I_{d}\right) .
$$

#### Predictor–Corrector（PC）混合采样

为了在逆向 SDE 采样中进一步提升样本质量，常引入 **Predictor–Corrector** 架构（类似NCSN中的退火LD）

1. **Predictor（预测器）**
    用一次 Euler–Maruyama 步更新逆向 SDE：

$$
\tilde{x}_{i-1}=x_{i}+\left[f-g^{2} s_{\theta}\right] \Delta t_{i}+g \sqrt{\Delta t_{i}} z_{i} .
$$

2. **Corrector（校正器）**
 在同一 $t_{i+1}$处对$\tilde x_{i+1}$做若干步 **Langevin Dynamics**（Unadjusted Langevin Algorithm）：

$$
x \leftarrow x+\frac{\eta}{2} s_{\theta}\left(x, t_{i+1}\right)+\sqrt{\eta} u, \quad u \sim \mathcal{N}(0, I) .
$$

以保证采样更好地“爬”到高密度区域。

实验结果证明，在确保计算资源充足的情况下，添加矫正步的好处大于单纯增加预测步。

#### 概率流 ODE（Probability Flow ODE）采样

> 对~~任意~~某些形式的（逆向）S-DE扩散过程（如加性噪声），存在一个对应的确定过程（ODE），其轨迹有相同的边缘概率分布${p_t(x)}_{t=0}^T$

与逆向 SDE 等价的确定性过程为
$$
\frac{\mathrm{d} x}{\mathrm{~d} t}=f(x, t)-\frac{1}{2} g(t)^{2} \nabla_{x} \log p_{t}(x) .
$$
在这个 ODE 中没有随机项，直接沿着学习到的梯度场收敛到数据分布。

再应用数值积分采样，如欧拉离散(与前向的离散方法一样)：
$$
x_{i-1}=x_{i}+\left[f\left(x_{i}, t_{i}\right)-\frac{1}{2} g\left(t_{i}\right)^{2} s_{\theta}\left(x_{i}, t_{i}\right)\right] \Delta t_{i} .
$$

## 流匹配 FLOW MATCHING FOR GENERATIVE MODELING

FLOW MATCHING FOR GENERATIVE MODELING, 2023-02-08, Yaron Lipman

### Continuous Normalization Flows

设样本$x$维度为$\mathbb R^d$，现在我们有一个简单的噪声分布$x_0\sim p_0(x)$，如高斯分布，我们想要生成一个数据分布$x_1\sim p_{\text{data}}$。

CNF方法会构造一个连续时间随机变量“**流(flow)**" (也称作transformation, map)，表示为$\phi(t,x_t):~[0,1]\times\mathbb R^d\to \mathbb R^d$ (时间参数写作下标)，要求其连续可微且可逆，逆函数亦连续可微
$$
x_t=\phi_t(x_0)
$$
它由ODE推动
$$
\frac{\mathrm d \phi_t(x_0)}{\mathrm d t}=v_t(\phi_t(x_0))\quad\text{or}\quad \frac{\mathrm d x_t}{\mathrm d t}=v_t(x_t)
$$
其中，$v_t:[0,1]\times \mathbb R^d\to \mathbb R^d$表示速度场。之前的方法是用神经网络$v_\theta(x_t,t)$学习速度场，进而建模一个CNF，即$\phi_\theta$，用以将简单分布$x_0$转换为目标分布$x_1$。可以套用以下"**push-forward equation**" (or change-of-variable rule)$p_t=[\phi_t]*p_0$来获得转换后变量的概率密度函数：
$$
p_t(x)=p_0(\phi^{-1}_t(x))\cdot\left|\det \frac{\part \phi^{-1}_t}{\part x} (x)\right|
$$
解释：若随机变量$Z\sim \pi(z)$，已知随机变量$X$与$Z$的关系:$X=f(Z)$，$f$可逆可导，则$X$的分布可以用以下公式求得：
$$
p(x)=\pi(z)\left|\frac{\mathrm dz}{\mathrm dx}\right|=\pi(f^{-1}(x))\left|\frac{\mathrm df^{-1}}{\mathrm dx}\right|=\pi (f^{-1}(x))\left|(f^{-1})^{'}(x) \right|
$$
将$\phi_t$视作$X_0\to X_t$的映射，已知$p_0$的分布，则新的随机变量$X_t$的分布（注意该映射是对原分布的所有点，而非某条轨迹）。而针对具体点：当你通过变换$\phi$得到一个样本$x$，则该样本在分布$p_t$上的概率可通过该点变换前在$p_0$分布上的概率乘以一个由变换雅可比决定的缩放系数获得。

同时，分布随时间的变化关系应满足一个守恒关系：（**Continuity Equation**）
$$
\frac{\partial p_{t}(x)}{\partial t}=-\nabla_{x} \cdot\left(p_{t}(x) v_{\theta}(x, t)\right)
$$
其中$\nabla_x\cdot()$表示相对于空间变量$x$的散度: $\sum_{i=1}^d\frac{\part}{\part x^{(i)}}$，该关系确保了速度场$v_t$”可生成“概率路径$p_t$（概率密度函数的性质需时刻满足）。

连续性方程描述密度的变化，但我们关心的是：当样本沿着 ODE 流动时，它的 log-density 如何随时间变化？

首先应用链式法则求$\log p_t(x_t)$对时间$t$求导，代入ODE


$$
\begin{aligned}
\label{eq:fm1}
\frac{d}{d t} \log p_{t}\left(x_{t}\right)&=\partial_{t} \log p_{t}(x)|_{x=x_t}+\nabla_{x} \log p_{t}(x)|_{x=x_t}^{\top} \frac{d x_{t}}{d t} \\
&=\partial_{t} \log p_{t}(x)|_{x=x_t}+\nabla_{x} \log p_{t}(x)|_{x=x_t}^{\top} v(x_t, t)
\end{aligned}
$$
第一项代入连续性方程，利用散度的性质：

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2025/11/18/20251118-162145.png" alt="image-20251118162145500" style="zoom:50%;" />
$$
\begin{aligned}
\partial_{t} \log p_{t}(x)&=-\frac{1}{p_t}\nabla_x\cdot(p_t(x)v_t(x))\\
&=-\frac{1}{p_t}\left[\nabla_xp_t(x)\cdot v_t(x)+\nabla_x\cdot (v_t(x)) \right]\\
&=-\nabla_x\log p_t(x)\cdot v_t(x)-\nabla_x\cdot (v_t(x))
\end{aligned}
$$
代回$\eqref{eq:fm1}$可得**instantaneous change of variable**：
$$
\frac{d}{d t} \log p_{t}\left(x_{t}\right)=-\nabla_x\cdot (v_t(x_t))
$$
终于，通过积分我们可以表示转换后的对数密度：
$$
\log p_{1}\left(x_{1}\right)=\log p_{0}\left(x_{0}\right)-\int_{0}^{1} \operatorname{div} v\left(x_{t},t\right) d t
$$
将其参数化
$$
\log p_{\theta}\left(x_{}\right)=\log p_{0}\left(x_{0}\right)-\int_{0}^{1} \operatorname{div} v_{\theta}\left(x_{t}, t\right) d t
$$
便可使用likelihood-based generative modeling方法$\Rightarrow \arg\max_\theta\mathbb E_{x\sim p_{\text {data}}}\log p_{\theta}\left(x_{}\right)$

---

总的来说，想要获得$\log p_{t}\left(x_{t}\right)$，需要求解以下联合ODE:
$$
\frac{\mathrm{d}}{\mathrm{~d} t}\binom{x_{t}}{\log p_{t}\left(x_{t}\right)}=\binom{v_{\theta}\left(t, x_{t}\right)}{-\operatorname{div} u_{\theta}\left(t, x_{t}\right)},
$$
这涉及对速度场$v$的散度计算，计算成本昂贵。一般要用 Hutchinson trace estimator。其他如高方差、训练慢、不稳定流等原因限制了CNF方法。**CNF 训练需要知道整个 path distribution $p_t$并从中采样，难度极高**。

### Flow Matching

流匹配是一种无需模拟即可训练的CNF方法，其直接针对向量场$v_\theta$构造一个回归目标函数:
$$
\label{eq:fm2}
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t \sim \mathcal{U}[0,1]} \mathbb{E}_{x \sim p_{t}}\left[\left\|v_{\theta}(t, x)-u(t, x)\right\|^{2}\right] .
$$
无需知道(求解)中间分布$p_t$。但如何获得一个有效的真实向量场$u_t$（满足连续性方程，即，能”生成“概率路径$p_t$）变成了主要问题。

> 对比flow matching的目标函数$\eqref{eq:fm2}$和score matching的目标函数$\eqref{eq:nscn1}$，两者十分相似。且groundtruth部分($u(t,x)\text{ and }\nabla_x\log p_{\text{data}}(x)$，包括DDPM中的逆扩散核$p(x_{t-1}|x_t)$)亦不可知。但NCSN将groundtruth替换为了$\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x)$，DDPM也选择了$q\left(x_{t-1} \mid x_t, x_0\right)$，即两者都选择添加目标(样本)数据作为condition，构造条件神经网络来学习。

### Conditional Flow Matching (CFM)

作者提出，与其寻找难以获得的真实$p_t$，不如人为构造一族 conditional probability path $p_t(x|x_1)$，其满足一些边界条件：

- $p_0(x|x_1)=q_0(x)$ 已知先验分布，如高斯
- $p_1(x|x_1)=\mathcal N(x;x_1,\sigma^2I)$，其中$\sigma$足够小，使得$x$的分布集中围绕于$x_1$，即样本数据点

边际化以上条件分布，即可获得边缘概率路径$p_t(x)$:
$$
\label{eq:cfm1}
p_{t}(x)=\int p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right) d x_{1}
$$
当$t=1$时，$p_1(x)$近似数据分布$p_{\text{data}}(x)$，下文中的$q(\cdot)$。

当然，我们还应当有个条件向量场 (**conditional vector field**) $u_t(x|x_1)$，以生成$p_t(x|x_1)$（满足连续性方程）。同样边际化该条件VF，获得边缘VF：
$$
\label{eq:cfm2}
u_{t}(x)=\int u_{t}\left(x \mid x_{1}\right) \frac{p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right)}{p_{t}(x)} d x_{1}
$$

>  式$\eqref{eq:cfm1}$好理解，局部概率路径对每个data point做加权求和（期望）就能得到一个全局的概率路径，但$\eqref{eq:cfm2}$又该如何解释？实际上该式等价于$u_{t}(x)=\mathbb{E}_{x_{1} \sim q\left(x_{1} \mid x\right)}\left[u_{t}\left(x \mid x_{1}\right)\right]$，针对后验概率$q\left(x_{1} \mid x\right)$做加权求和，而该后验的意义是”在$x_t=x$这个位置，你可能来自哪些路径($x_1$)，它们各自给出的速度是多少？“。
>
> 假设你在空间中某个位置 x，时刻 t，积分对象是可能流向的 data points，权重为流向每个 data point 的概率，平均后的全局向量场应该等于$x$处**所有可能目的地的平均方向**。

**定理一**：若条件向量场$u_t(x|x_1)$可生成条件概率路径$p_t(x|x_1)$，则边缘化后的$u_t \eqref{\label{eq:cfm1}}$可生成边缘化的$p_t(x)\eqref{\label{eq:cfm2}}$。

证明：只需证明$u_t,~p(t)$满足连续性方程
$$
\begin{aligned}
\frac{d}{d t} p_{t}(x) & =\int\left(\frac{d}{d t} p_{t}\left(x \mid x_{1}\right)\right) q\left(x_{1}\right) d x_{1}\\
&=-\int \operatorname{div}\left(u_{t}\left(x \mid x_{1}\right) p_{t}\left(x \mid x_{1}\right)\right) q\left(x_{1}\right) d x_{1} \\
& =-\operatorname{div}\left(\int u_{t}\left(x \mid x_{1}\right) p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right) d x_{1}\right)\\
&=-\operatorname{div}\left(u_{t}(x) p_{t}(x)\right)
\end{aligned}
$$
条件化的优势是：我们只需构造局部的path $p_t(\cdot|x_1)$，如线性、高斯路径，而非整个$p_t$，且可以从中采样。

定理一告诉我们，构造条件路径其实等价于构造全局路径。

---

既然引入条件路径，那目标函数也改变为：
$$
\label{eq:cfmloss}
\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t \sim \mathcal{U}[0,1], x_{1} \sim q, x_{t} \sim p_{t}\left(x \mid x_{1}\right)}\left[\left\|v_{\theta}(t, x)-u_{t}\left(x \mid x_{1}\right)\right\|^{2}\right] .
$$
$v_{\theta}(t, x)$为神经网络用于近似条件向量场。

**定理二**：损失函数$\mathcal L_{FM} \eqref{eq:fm2}$与$\mathcal L_{CFM} \eqref{eq:cfmloss}$两者等价，只相差一个与$\theta$无关的常数。

证明：展开
$$
\begin{aligned}
\left\|v_{\theta}(t,x)-u_{t}(x)\right\|^{2} & =\left\|v_{\theta}(t,x)\right\|^{2}-2\left\langle v_{\theta}(t,x), u_{t}(x)\right\rangle+\left\|u_{t}(x)\right\|^{2} \\
\left\|v_{\theta}(tx)-u_{t}\left(x \mid x_{1}\right)\right\|^{2} & =\left\|v_{\theta}(t,x)\right\|^{2}-2\left\langle v_{\theta}(t,x), u_{t}\left(x \mid x_{1}\right)\right\rangle+\left\|u_{t}\left(x \mid x_{1}\right)\right\|^{2}
\end{aligned}
$$

- 第一项：$\mathcal L_{11}=\int p_t(x)\left\|v_{\theta}(t,x)\right\|^{2} \mathrm dx=\int p_t(x,x_1)\mathrm dx_1\left\|v_{\theta}(t,x)\right\|^{2} \mathrm dx=\int p_t(x|x_1)q(x_1)\mathrm dx_1\left\|v_{\theta}(t,x)\right\|^{2} \mathrm dx=\mathcal L_{21}$ 
- 第二项：

$$
\begin{aligned}
\mathbb{E}_{x \sim p_{t}}\left\langle u_{\theta}(t, x), u_{t}(x)\right\rangle & =\int\left\langle u_{\theta}(t, x), \int u_{t}\left(x \mid x_{1}\right) \frac{p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right)}{p_{t}(x)} d x_{1}\right\rangle p_{t}(x) \mathrm{d} x \\
& =\int\left\langle u_{\theta}(t, x), \int u_{t}\left(x \mid x_{1}\right) p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right) d x_{1}\right\rangle \mathrm{d} x \\
& =\iint\left\langle u_{\theta}(t, x), u_{t}\left(x \mid x_{1}\right)\right\rangle p_{t}\left(x \mid x_{1}\right) q\left(x_{1}\right) d x_{1} \mathrm{~d} x \\
& =\mathbb{E}_{q_{1}\left(x_{1}\right) p\left(x \mid x_{1}\right)}\left\langle u_{\theta}(t, x), u_{t}\left(x \mid x_{1}\right)\right\rangle
\end{aligned}
$$

- 第三项与$\theta$无关

定理二给出了新的可行的训练目标。

---

有了以上目标函数，下一步就是获得$u_{t}\left(x \mid x_{1}\right)$的表达式了。

现在让我们构建一类高斯条件概率路径：
$$
p_{t}\left(x \mid x_{1}\right)=\mathcal{N}\left(x ; \mu_{t}\left(x_{1}\right), \sigma_{t}\left(x_{1}\right)^{2} \mathrm{I}\right)
$$
同样要满足边界条件：

- $t=0,~p_0(x|x_1)=\mathcal N(x;0,I)\Rightarrow \mu_0(x_1)=0,~\sigma_0(x_1)=1$
- $t=1,~p_1(x|x_1)=\mathcal N(x;x_1,\sigma_{\min}^2)\Rightarrow \mu_1(x_1)=x_1,~\sigma_1(x_1)=\sigma_{\min}\to 0$

**概率密度的演化没有唯一对应的速度场，速度场本身可以非常多（无穷多）**。作者选择
$$
\psi_{t}(x_0;x_1)=\sigma_{t}\left(x_{1}\right) x_0+\mu_{t}\left(x_{1}\right) .
$$
这种最简单的可逆仿射变换作为流的映射方程。可证明，该变换满足前推方程：$[\psi_t]*p_0(x|x_1)=p_t(x|x_1)$

> 根据CNF的前置介绍，连续流的映射是将$x_0$映射到$x_t$，这里$x_1$作为条件（依赖）；同理，下面的条件向量场中自变量应为t时刻的空间坐标，即$x_t$。

**定理三**：通过以上构筑，条件向量场的形式为：
$$
u_{t}\left(x \mid x_{1}\right)=\frac{\sigma_{t}^{\prime}\left(x_{1}\right)}{\sigma_{t}\left(x_{1}\right)}\left(x-\mu_{t}\left(x_{1}\right)\right)+\mu_{t}^{\prime}\left(x_{1}\right) .
$$
证明：由向量场定义，对t求偏导：
$$
\partial_{t} \psi_{t}\left(x_{0} ; x_{1}\right)=\dot{\sigma}_{t}\left(x_{1}\right) x_{0}+\dot{\mu}_{t}\left(x_{1}\right) .
$$
由映射关系$x_t=\psi_{t}(x_0;x_1)=\sigma_{t}\left(x_{1}\right) x_0+\mu_{t}\left(x_{1}\right)$写出其逆映射（这一步单纯因为条件仅 over $x_1$）：
$$
x_{0}=\frac{x_{t}-\mu_{t}\left(x_{1}\right)}{\sigma_{t}\left(x_{1}\right)} .
$$
代回可得：
$$
u_{t}\left(x_{t} \mid x_{1}\right)=\dot{\sigma}_{t}\left(x_{1}\right) \frac{x_{t}-\mu_{t}\left(x_{1}\right)}{\sigma_{t}\left(x_{1}\right)}+\dot{\mu}_{t}\left(x_{1}\right) .
$$
将自变量$x_t$替换为$x$（符号替换），即得定理中的表达式。

---

### 一些实例化

上面我们定义了条件概率路径的形式$p_{t}\left(x \mid x_{1}\right)=\mathcal{N}\left(x ; \mu_{t}\left(x_{1}\right), \sigma_{t}\left(x_{1}\right)^{2} \mathrm{I}\right)$，而$\mu_t,\sigma_t$具体的实例化又该如何呢？

#### Optimal Transport conditional VFs

作者给出了一种实例化方案：**Optimal Transport conditional VFs**

对边界条件做线性插值（linear interpolation）：
$$
\mu_{t}\left(x_{1}\right) \triangleq t x_{1} \quad \text { and } \quad \sigma_{t}\left(x_{1}\right) \triangleq(1-t)+t \sigma_{\min }
$$
该设置满足边界条件，其对时间的导数为：
$$
\dot{\mu}_{t}\left(x_{1}\right) \triangleq x_{1} \quad \text { and } \quad \dot{\sigma}_{t}\left(x_{1}\right) \triangleq-1+\sigma_{\min }
$$
带入定理三的公式中，得到条件速度场如下：
$$
u_{t}\left(x \mid x_{1}\right)=\frac{x_{1}-\left(1-\sigma_{\min }\right) x}{1-\left(1-\sigma_{\min }\right) t}
$$
对应的条件流写作：
$$
\psi_{t}(x)=\left(1-\left(1-\sigma_{\min }\right) t\right) x+t x_{1},
$$
损失函数为：
$$
\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q\left(x_{1}\right), p\left(x_{0}\right)}\left\|v_{\theta}\left(\psi_{t}\left(x_{0},t\right)\right)-\left(x_{1}-\left(1-\sigma_{\min }\right) x_{0}\right)\right\|^{2} .
$$


#### 与SMLD/DDPM的关系

在SDE的框架下，我们总结了SMLD的前向SDE是一个Variance Exploding (VE) SDE，所以概率路径可以设为：
$$
p_{t}(x)=\mathcal{N}\left(x \mid x_{1}, \sigma_{1-t}^{2} I\right)
$$
初始时刻方差$\sigma_{1}\gg 1$分布近似纯高斯，$t=1$时$\sigma_0=0$。

同样，DDPM的前向过程是Variance Preserving (VP)的，概率路径设为：
$$
p_{t}\left(x \mid x_{1}\right)=\mathcal{N}\left(x \mid \alpha_{1-t} x_{1},\left(1-\alpha_{1-t}^{2}\right) I\right), \text { where } \alpha_{t}=e^{-\frac{1}{2} T(t)}, T(t)=\int_{0}^{t} \beta(s) d s
$$
均值系数$\alpha_{1-t}$由0到1递增，方差由1到0递减。

### 训练与采样

训练伪代码

```python
# Hyperparams: batch_size, lr, n_steps, sigma_min, p0 ~ N(0,I), data q(x1)
# Model: u_theta(t, x) -> R^d  (takes time scalar t and space x)
# time embedding: embed(t) -> vector

for step in range(n_steps):
    # 1) sample minibatch
    x1_batch = sample_minibatch_from_data(batch_size)      # x1 ~ q(x1)
    x0_batch = sample_normal(batch_size)                   # x0 ~ p0 = N(0,I)
    t_batch  = sample_uniform(0, 1, batch_size)            # t ~ U(0,1)

    # 2) compute sigma_t and mu_t
    sigma_t = 1 + t_batch * (sigma_min - 1)                # shape [B,1]
    mu_t    = t_batch[:,None] * x1_batch                   # broadcast -> [B,d]

    # 3) compute x_t (the point at which to evaluate model)
    x_t = sigma_t[:,None] * x0_batch + mu_t                # [B,d]

    # 4) compute target u_t(x_t | x1)
    # dot_sigma is constant = sigma_min - 1
    target = ((sigma_min - 1) / sigma_t)[:,None] * (x_t - mu_t) + x1_batch

    # 5) predict with model
    pred = u_theta(t_batch, x_t)                           # [B,d]

    # 6) loss and update (MSE)
    loss = mean( ||pred - target||^2 over batch and dims )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

- **Unconditional FM（论文中常用的“FM”）**：模型$v_\theta(t,x)$**不依赖**$x_1$。
- **Conditional FM（CFM）**：模型$f_\theta(t,x\mid x_1)$**显式接收**$x_1$作为条件并直接拟合$u_t(x\mid x_1)$。但采样仍需要无条件$u_\theta$

---

求解ODE获得$x_t=\phi_t(x_0)=\int_0^1v_\theta\mathrm dt$

采样伪代码（使用显式 RK solver 或黑盒 ODE solver）：

```python
# sampling params: n_steps_S, solver step-size dt
# Starting distribution p0 = N(0,I)

for i in range(n_samples):
    x = sample_normal()           # x at t=0

    # integrate from t=0 to t=1
    t = 0.0
    while t < 1.0:
        # one RK4 step (example)
        dt = min(dt_max, 1.0 - t)
        k1 = u_theta(t, x)
        k2 = u_theta(t + dt/2, x + dt/2 * k1)
        k3 = u_theta(t + dt/2, x + dt/2 * k2)
        k4 = u_theta(t + dt, x + dt * k3)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt

    x_sample = x   # at t=1, approximately ~ data distribution
    save(x_sample)

```

## Rectified Flow

Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, 2022-09, Liu ==（早于FM）==

### 动机

首先简要回顾FM的思想，方便与后面介绍的RF比较:

给定先验分布$p_0(x)$（通常是高斯），目标数据分布$p_1(x)$。概率路径$p_t$由一个连续时间映射$x_t=\phi_t(x_0)$决定。而该映射（或“流”）由一个ODE驱动：$\frac{\mathrm d \phi_t(x_0)}{\mathrm d t}=v_t(\phi_t(x_0))$，其中$v_t$被称为速度场

FM的想法是用回归的方式学习$v_{\theta}(t,x_t)$。为了能够采样groundtruth做回归训练，CFM引入了对目标数据点的条件化，主动构造了一类高斯条件概率路径$p_{t}\left(x \mid x_{1}\right)=\mathcal{N}\left(x ; \mu_{t}\left(x_{1}\right), \sigma_{t}\left(x_{1}\right)^{2} \mathrm{I}\right)$以及对应的线性插值映射$\psi_{t}(x_0;x_1)=\left(1-\left(1-\sigma_{\min }\right) t\right) x_0+t x_{1}$，其中$\mu_{t}\left(x_{1}\right) \triangleq t x_{1} \quad \text { and } \quad \sigma_{t}\left(x_{1}\right) \triangleq(1-t)+t \sigma_{\min }$。然后让模型取学习相应的条件速度场$u_{t}\left(x \mid x_{1}\right)$。



回顾采样过程$x_{t+\Delta t}=x_t+\Delta t\cdot v_t$，既然最终学习的向量场是为了驱动分布$p_0$上的样本点到分布$p_1$上，那为什么不直接走直线呢？所以RF直接对速度场的构造下手

### Rectified Flow

用线性插值构造样本对$(x_0,x_1)$之间的flow：
$$
\psi_t(x_0,x_1)=(1-t)x_0+tx_1
$$
如何选择样本对？选定一个耦合（joint coupling）$\pi(x_0,x_1)$满足 marginals 分别为$p_0,p_1$。（常见简单选法是独立耦合：先验样本与真实样本独立配对$\Rightarrow \pi(x_0,x_1)=p_0(x_0)p_1(x_1)$）

根据定义，路径上的瞬时速度（对参数对$x_0,x_1$ 固定时），即速度场：
$$
\label{eq:rf1}
u(x_0,x_1)=x_1-x_0
$$
注意，在直线插值下$u$不再与$t$相关（恒定沿路径$\psi:x_0\to x_1$）。

但以上设定有个bug：我们构建向量场来引导$x_0$向$x_1$变换，但这个向量场却需要已知$x_1$，采样(模拟)过程违背因果关系。且由于任意配对，在时刻$t$，$x_t = \psi_t(x_0,x_1)$可能来自多个$(x_0,x_1)$样本对之间的轨迹点（intersect），速度关于$x_t$不再是确定的，而是基于一个潜变量条件$z=(x_0,x_1)$，换句话说，速度场$\eqref{eq:rf1}$实际上可写作双边条件速度场（**two-sided conditional vector field**）$u_t(x_t\mid z)$，轨迹点的速度由$(x_0,x_1)$配对决定。这是RF与之前介绍的单边CMF的一大不同。

>![Realizations of conditional paths following conditional vector field $u_t(x \mid x_1)$ from \eqref{eq:mog2mog}. Paths are highlighted by the sign of the 2nd vector component.](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2025/11/21/20251121-140925.png)
>
>左边部分(包括上下)为初始分布$X_0$，右边是目标分布$X_1$，可以看到不同$(x_0,x_1)$对之间的轨迹有交叉，即存在$x_t$有多个$x_1$，其速度向量无法确定，需要（对不同的$x_1$）做均值以估计该位置的速度方向。独立随机的配对让梯度估计的方差变大，训练难以收敛。这是任何CFM训练时的情况，不同的流设定可能会让连接的“线”曲度不同，而正是RF的直线插值让相连的线“变直”。





避开因果悖论的方法就是引入神经网络$v_\theta(x_t,t)$去学习这个$u(x_0,x_1)$真实速度场，因为后者在训练时是可采样（计算）的。更一般的CMF损失函数表示为：
$$
\mathcal{L}_{\text{CMF}}(\theta)=\mathbb{E}_{t \sim \mathcal U[0,1], z \sim \pi(z), x_t\sim p_t(x\mid z)}\left[\left\|v_{\theta}\left(x_{t}, t\right)-u_t\left(x_t\mid z\right)\right\|^{2}\right] .
$$
而RF正是一种特别的实例化，其损失函数为：
$$
\mathcal{L}_{\text{RF}}(\theta)=\mathbb{E}_{t \sim \mathcal U[0,1], x_0 \sim p_0,x_1\sim p_1, x_t\sim p_t(x\mid x_0,x_1)}\left[\left\|v_{\theta}\left(x_{t}, t\right)-(x_1-x_0)\right\|^{2}\right] .
$$
直观上：在时间$t$我们观察到一个位置$x=x_t$，满足哪个速度最“合理”？最合理的是所有生成这个$x$的路径速度的条件均值。

---

由于这个特殊的对时间$t$恒定的速度场，根据原论文解释，模型学的其实是**路径速度的条件期望**：
$$
v^{*}(x, t)=\mathbb{E}\left[u\left(x_{0}, x_{1}\right) \mid x_{t}=x\right]=\mathbb{E}\left[x_{1}-x_{0} \mid x_{t}=x\right]=\arg\min_v\int_0^1\mathbb E_{\pi(x_0,x_1)}\left[ \|x_1-x_0-v(x_t,t) \|^2\right] .
$$
虽然在独立耦合下：

- 同一个位置$x$可以由无数个不同对$(x_0,x_1)$生成
- 它们各自的速度$u=x_1-x_0$完全不同
- 各条直线路径是会相交的

但对 **所有到达这个点的路径速度求平均** 之后，$v^*(x,t)$就变成了一个 **单一值**。也就是说：

- “路径混乱”
   → 在条件化后被“平均掉”
- “速度不一致”
   → 经过期望成为“最一致方向”

因此模型学到的是：

> **经过所有可能路径平均后，在这个点上最合理的平均流动方向。**

事实上，由于ODE的解具有唯一性，采样的路径并不会出现相交（无论选择什么条件概率路径/速度场），自动“rewire”。RF做的只是让路径“更直”，方向更“确定”。

> ![Realizations of marginal paths following the marginal vector field $u_t(x)$ from \eqref{eq:mog2mog}. Paths are highlighted by the sign of the 2nd vector component.](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2025/11/21/20251121-194315.png)
>
> 用学习的$v^*$求解ODE采样时的路径。ODE仍然保留了原来的基本路径，但是做了一个重组来避免相交的情况。

总结一下：RF 中的 “Rectification” 不是在“修正训练时路径的交叉”，而是将任意耦合下的最优路径族，规整为直线插值 (把复杂轨迹“矫正（rectify）”成直线 )，使得模型必须逼近一个低曲率、方向一致的速度场: $\frac{d^{2} \psi_{t}}{d t^{2}}=0$。低曲率 → 速度场稳定 → ODE 非 stiff → 步长巨大、可几步生成

### 讨论一些特性

**在模型容量充足且最小化到全局最小值时**，我们学到的速度场就是$v^*(x,t)$。用ODE采样获得的轨迹表示为$Z_t$。

**特性1**：ODE轨迹$Z_t$和直线插值$X_t$拥有相同的边缘分布，即：
$$
\text{Law}(Z_t)=\text{Law}(X_t),\quad \forall t\in [0,1]
$$
因此，$(Z_0,Z_1)$可视作是$\pi_0$和$\pi_1$的一种耦合。

**特性2**：$(Z_0,Z_1)$相比$(X_0,X_1)$能保证一个更小或至少相等的传输损失（可理解为连接路径的长度），且这是对所有凸损失函数有效：
$$
\mathbb{E}\left[c\left(Z_{1}-Z_{0}\right)\right] \leq \mathbb{E}\left[c\left(X_{1}-X_{0}\right)\right], \quad \forall \text { convex } c: \mathbb{R}^{d} \rightarrow \mathbb{R}
$$
这一点可以有如下解释：数据对$(X_0,X_1)$可以是$\pi_0,\pi_1$的任意耦合，如独立耦合，而经RF矫正的耦合是由ODE产生的确定性耦合。

**特性3**：ReFlow操作：从$(Z_0^{0},Z_1^0)=(X_0,X_1)$开始，递归的应用矫正后的耦合来生成新的流：
$$
\mathbb E_{z^k_0\sim \pi_0,z^k_1\sim \text{Flow}^k(z_1^k|z_0^k)}
$$
其中，$z_1^k$是模拟ODE采样得到的，重复Reflow多次，可以不断降低传输损失，从而将flow拉得更直。

> 实际中，因为每次$v$优化得不完美，多次Reflow会积累误差，所以我们不建议做太多次的Reflow。

官方可视化例子：https://colab.research.google.com/drive/1g8Fm_S4BqrDaG2eHI3sDulBD3WbK2D_V?usp=sharing#scrollTo=l5SBh-_FTo3G

作者还讨论了RF的非线性扩展：

对于之前的概率流ODE（即逆过程），包括VE, VP, sub-VP ODE，可以在矫正流的框架下统一：
$$
X_{t}=\alpha_{t} X_{1}+\beta_{t} \xi, \text { with } \xi \sim \mathcal{N}(0, I)
$$
只是流轨迹矫正的方向不再是直线。前面FM介绍过，这里不再赘述。

### RF与FM关系以及一些注意点

在FM中给的OT path，即 $\psi_{t}(x_0;x_1)=\left(1-\left(1-\sigma_{\min }\right) t\right) x_0+t x_{1}$中，当$\sigma_1=\sigma_{\min}$取0时，路径就与RF完全一致。但FM的速度场构造公式不再成立，因为用到了$\sigma_t$的倒数，且映射的反函数也不成立，当$t\to1$时，会导致目标速度发散，损失项出现梯度爆炸，ODE 速度场不可数值稳定地学习。FM 使用 CNF 的参数化，需要保持密度连续、可微、有界，需要$\sigma_{\min}>0$，因此 FM 的公式体系并不能直接退化到 RF 的公式体系。FM论文比RF晚，其给出了更general的meta-framework



**几处重要的工程/理论注意点**

- **耦合 $\pi$**：独立耦合（从$p_0$与$p_1$独立采样）是最简单的选择。更“聪明”的耦合（例如接近最优输运的耦合）可能使得条件期望更简单、更容易拟合。
- **时间采样权重**：在实践中，有时在损失中对不同 t 赋予权重（例如弱化靠近边界 t≈0/1 的采样）能提升稳定性。
- **模型容量与正则**：因为我们直接训练速度值，模型需要对不同 t 的速度作出细致建模；网络架构和正则化（例如谱归一化、层归一化、时间编码）非常关键。
- **数值积分（采样）**：由于 RF 学到的是确定性 ODE，使用高阶 ODE solver（Heun, RK4）通常能在很少步数下给出高质量样本；这也是 RF 采样快的原因之一。
- **边界行为**：t → 0 或 1 时若路径和耦合导致密度比产生数值问题（例如极端缩放），可能需要对时间做 reparameterization 或加噪保证稳定。

### 参考链接

https://zhuanlan.zhihu.com/p/603740431

https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html

