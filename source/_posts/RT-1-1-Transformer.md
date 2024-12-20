---
title: RT-1(一)——从维度变化角度学习Transformer架构
date: 2024-12-20 17:33:24
tags:[Transformer-based, Deep learning, 算法实现]
categories: Generative Models
---

# RT-1(一)——从维度变化角度学习Transformer架构

## Transformer神经网络架构详解

背景：为处理语言序列的翻译任务，Google在2017年的《Attention Is All You Need》论文中提出Transformer模型。

其模型架构如图所示：

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-040650.png" alt="img" style="zoom:50%;" />

```rust
输入序列 --> [嵌入层 + 位置编码] --> 编码器层（多头注意力 + 前馈网络） --> 解码器层（多头注意力 + 编码器-解码器注意力 + 前馈网络） -->[线性层 + Sofmax层] --> 输出序列
```

1. 输入嵌入层  Input Embedding + 位置编码 Positional Encoding
2. 编码组件 Encoding component
   - 多层编码器 Encoder x 6
     - 多头注意力机制 Multi-head Attention Mechanism
       - 自注意力机制 Self-Attention Mechanism
     - 位置前馈网络 Position-wise Feed-Forward Networks
     - 残差连接和层归一化 residual connection & layer-normalization
3. 解码组件 Decoding component
   - 多层解码器 Decoder x 6
     - Masked Multi-head Attention 
     - Encoder-Decoder Attention Layer
     - FNN
4. 线性层 + Sofmax层

### 输入嵌入

将长度为$N$的**输入序列**中每个Mark: $m_i\in\mathbb R^V$（从词汇表中选择，tokenize+one-hot编码+矩阵乘法？）都映射成一个**嵌入向量**：$e_i=W[m_i]$，其中$W\in\mathbb R^{V\times d_{model}}$为嵌入矩阵，$e_i\in\mathbb R^{d_{model}}$包含**语义信息**，$V$为词汇表大小，$d_{model}$为模型隐藏层维度。

故对应输出应为一组向量，序列长度$N$一般由训练数据中最长的句子决定（不足则padding）。按原论文，往下设$d_{model}=512$。

### 位置编码

为获取输入序列的位置信息，为每个嵌入向量加上一个包含**位置信息**的**位置编码向量**$p_i\in\mathbb R^{d_{model}}$，组成最终的**输入表示**: $x_i\in\mathbb R^{d_{model}}$

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-053311.png" alt="img" style="zoom:50%;" />

位置编码实现：<https://blog.csdn.net/xian0710830114/article/details/133377460>

基于正弦和余弦函数的固定位置编码方式：

对于序列中位置为 $i$的token ，其位置编码向量中第 $j$ 个维度元素计算公式为：
$$
PE^i_j=\left\{\begin{matrix}
\sin \left(\frac{i}{10000^{2 k / d_{\text {model }}}}\right)  &,  &j=2k \\
\cos \left(\frac{i}{10000^{2 k / d_{\text {moded }}}}\right)  &,  &j=2k+1
\end{matrix}\right.,\quad \text{where }k=0,1,2,\cdots \text{indicate the idx of pair.}
$$
写成向量形式为：
$$
p_i=\begin{bmatrix}\sin(i\cdot\omega_0)\\
\cos(i\cdot\omega_0)\\
\sin(i\cdot\omega_1)\\
\cos(i\cdot\omega_1)\\
\vdots\\
\sin(i\cdot\omega_{d_{model}/2-1})\\
\cos(i\cdot\omega_{d_{model}/2-1})\\
\end{bmatrix}\in\mathbb R^{d_{model}}.
$$
where $\omega_k=\frac{1}{10000^{2 k / d_{\text {model }}}}\in[\omega_{d_{model}/2-1},\omega_0]\subseteq(1/10000,1]$.

如此，不同位置的token在同一维度上的位置编码值可视作同一频率的正弦波，所有元素数值在$[-1,1]$之间。

**特点**：

1. **周期性**：正弦和余弦函数的周期性允许模型感知相对位置信息。
2. **无参数性**：位置编码是固定生成的，不需要训练额外的参数。
3. **可扩展性**：可以适用于任意长度的输入序列。

### 编码器

每个编码器均由自注意力层(self-attention layer)和位置前馈网络(Position-wise Feed Forward Network, FFN)组成。

#### 自注意力机制

**自注意力机制**（Self-Attention）是 Transformer 架构的核心模块，用于捕获输入序列中不同位置之间的全局依赖关系。与传统的卷积或递归操作不同，自注意力机制可以直接建模序列中任意两个位置的交互，从而实现对全局信息的高效提取。

当模型处理每个词（输入序列中的每个位置）时，Self-Attention 机制使得模型不仅能够关注当前位置的词，而且能够关注句子中其他位置的词，从而可以更好地编码这个词。

---

**具体实现**

1. 计算每个输入向量$\mathrm{x}_i$的查询(Query)、键(Key) 和值( Value) 向量：$\rm q_i,\rm k_i,\rm v_i\in \mathbb R^{d_k}$: (往下设$d_k=64$)

$$
\mathrm{q}_i = x_i^\top W^Q, \quad \mathrm{k}_i = x_i^\top W^K, \quad \mathrm{v}_i = x_i^\top W^V,
$$

其中$W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_k} $是可训练的**权重矩阵**。

2. 计算某个输入向量对句子中其它向量的注意力分数: (dot product)

$$
\text{scores}_j^i = q_i\bullet k_j
$$

> Q：自己的研究方向； K：别人论文的关键词；V：论文的具体内容

这里，$\text{scores}_j^i$ 表示序列中第 $i$ 个位置对第 $j$ 个位置的关注程度。

3.  归一化注意力分数

为了让注意力分数更稳定且易于解释，对分数进行缩放和归一化：
$$
\text{scaled scores}^i_j = \frac{\text{scores}^i_j}{\sqrt{d_k}}, \quad \text{attention weights}^i = \text{softmax}\left(\text{concat}_j(\text{scaled scores}^i_j)\right)
$$


- 缩放因子 $\sqrt{d_k}$ 避免点积值过大，导致梯度不稳定。
- `Softmax` 将分数归一化为概率分布。

4. 加权求和生成输出

使用相应的注意力权重 $\text{attention weights}^i_j $对值向量 $\mathrm{v}_i$作 加权求和，生成最终的自注意力输出：
$$
\mathrm z^i =\sum_j \text{attention weights}^i_j \cdot \mathrm v_j
$$


输出 $\mathrm z_i \in \mathbb{R}^{d_k} $是序列中每个位置的上下文向量，结合了全局信息。

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-204833.png" alt="img" style="zoom:50%;" />

5. 直接用矩阵计算：

   1. 计算$Q,K,V$矩阵：$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$, where $X\in\mathbb R^{N\times d_{model}}=\mathbb R^{10\times 512}$ is Embedding Matrix, $Q, K, V \in \mathbb{R}^{N \times d_k}=\mathbb{R}^{10 \times 64}$.
   2. 计算自注意力：$Z=\text{softmax}(\frac{QK^\top}{\sqrt {d_k}})\cdot V$, where $Z\in\mathbb R^{N\times d_{k}}=\mathbb R^{10\times 64}$.

   

#### 多头自注意力机制

对单头注意力的进一步完善。

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-210330.png" alt="img" style="zoom:50%;" />

**具体实现**

1. 常规方法：用$h$组独立的权重矩阵($W_i^Q,W_i^K,W_i^V$，均随机初始化)分别计算$h$组注意力矩阵$Q_i,K_i,V_i$，然后分别计算各个子空间的自注意力:
   $$
   Z_i = \text{Attention}(Q_i, K_i, V_i)\in \mathbb R^{N\times d_{k}}
   $$
   

   往下设$h=8$

2. 将所有头的输出沿第二个维度拼接在一起，并通过一个线性变换映射回原空间维度：
   $$
   Z = \text{Concat}(Z_0, \dots, Z_{h-1})\cdot W^O\in \mathbb R^{N\times d_{model}}
   $$
   

   - $W^O \in \mathbb{R}^{h\cdot d_k \times d_{model}}= \mathbb{R}^{512 \times 512}$是可训练投影矩阵，作用是将拼接结果映射回原始维度。
   - 注意$d_k(=d_v)=d_{model}/h$

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-212120.png" alt="img" style="zoom:50%;" />

> 代码实现中，通过设计$d_k(=d_v)=d_{model}/h$，无需显示地将$Q,K,V$投影到$h$个维度为$\mathbb R^{d_k}$子空间，而是直接通过一组$W^Q,W^K,W^V\in \mathbb R^{d_{model}\times d_{model}}$直接将嵌入矩阵$X$投影到$d_{model}$维度的全空间中，再通过reshape操作将$d_{model}$维度转化为$h\times d_k$，即实现了$h$组注意力矩阵的划分。如此无需增加过大的额外参数数量，而拓展了模型在不同子空间下对不同位置关注度的能力，再通过最后可训练的投影权重综合有价值的关注点，可以有效防止过拟合。

#### 位置前馈网络

由两个全连接层以及中间的ReLU激活函数组成：
$$
\text{FFN}(x)=\max(0,x\mathrm W_1+b_1)\mathrm W_2+b_2
$$
输入输出维度为:$d_{model}=512$，隐藏层维度为$d_{ff}=2048$，一般为4倍大小关系。

#### 残差连接和层归一化

每个编码器/解码器的每个子层（Self-Attention 层和 FFN 层，以及解码器特有的Encoder-Decoder Attention层）都有一个残差连接，再执行一个层标准化操作，整个计算过程可以表示为：
$$
X_{residual}=X_{in}+X_{out},\\
X_{norm}=\text{LayerNorm}(X_{residual}).
$$
<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-215945.png" alt="img" style="zoom: 67%;" />

### 解码器

最后一个编码器输出是一个二维张量$X_{\text{encoder-out}}\in\mathbb R^{N\times d_{model}}$(一组注意力矩阵$K$和 $V$ <mark>(半个自注意力层结尾 ?)</mark>，**修正**)，输入到每个解码器的Encoder-Decoder Attention层(即图1中的Multi-Head Attention，又称交叉注意力模块)计算出对应的注意力矩阵$K_X$和 $V_X$。

~~此外，第一个解码器还接收$(y_1,\cdots,y_t)$(<mark>(最新的完整的，还是单单上一个时间步输出 ? 后者)</mark>, **修正**)输出序列(或**目标序列**，长度为$M$)(需要Embedding->$\mathbb R^{M\times d_{model}}$)作为输入，输出新的预测$(\hat y_2,\cdots,\hat y_{t+1})$。~~

每个时间步的输出序列(或**目标序列**，长度为$M$)将作为下一个时间步下解码器的输入(需经过嵌入和位置编码)，表示为$Y\in\mathbb R^{M\times d_{model}}$，随后经过一个**掩码多头自注意力层**计算目标序列内部的注意力: $Z_Y\in\mathbb R^{M\times d_{k}}$。（具体看代码实现-训练和推理部分）

![img](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-222217.png)

#### 交叉注意力模块 Encoder-Decoder Attention

Encoder-Decoder Attention 层的工作原理和多头自注意力机制类似。不同之处是：Encoder-Decoder Attention 层使用前一层的输出构造 Query 矩阵，而 Key 和 Value 矩阵由编码器栈的输出构造。目的是将目标序列的隐藏表示与编码器的输出进行交互，捕获目标序列与源序列之间的上下文关系。

维度变化如下：

1. 由编码器栈的输出$X_{\text{encoder-out}}\in\mathbb R^{N\times d_{model}}$构造注意力矩阵$K_X$和 $V_X$，维度为$\mathbb R^{N\times d_k}$
2. 由解码器上一子层的输出$Y_{\text{sublayer-out}}\in\mathbb R^{M\times d_{model}}$构造注意力矩阵$Q_Y$，维度为$\mathbb R^{N\times d_k}$
3. 线性映射到$h$个子空间，分别计算各头的交叉注意力：$Z_i=\text{softmax}(\frac{Q_YK_X^\top}{\sqrt {d_k}})\cdot V_X\in\mathbb R^{M\times d_{k}}$
4. 各头注意力拼接再线性映射回原维度空间，获得最终的交叉注意力$Z_{XY}\in\mathbb R^{M\times d_{model}}$

**解码器推理演示动画：**

![img](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/12/16/20241216-222452.gif)

> 辨析自注意力与交叉注意力：
>
> - 自注意力指序列自身对自身的注意力计算，其发生在编码器和解码器的第一层
> - 交叉注意力(又或Encoder-Decoder注意力)指的是目标(嵌入)序列对输入(嵌入)序列的注意力，其发生在解码器第三层

### 最后的线性层和Softmax层

最后一个解码器输出的$Y_{\text{encoder-out}}\in \mathbb R^{M\times d_{model}}$中，每个子向量$y_i\in \mathbb R^{d_{model}}$将通过线性层(全连接层)被映射到更高维的向量空间中($\mathbb R^V$取决于词汇表大小)：
$$
Y_{\text{logits}}=Y_{\text{encoder-out}} W_{\text{out}}+b_{\text{out}}\in \mathbb R^{M\times V},\quad\text{where } W_{\text{out}}\in\mathbb R^{d_{model}\times V},~b_{\text{out}}\in\mathbb R^V
$$
再由`softmax`将各个词的分数转为概率，最后选取概率最大的词作为该时间步的输出。

### 其它细节

#### 两个嵌入层以及最后的线性层

在 Transformer 论文，提到一个细节：编码组件和解码组件中的嵌入层，以及最后的线性层共享权重矩阵 (?)。不过，在嵌入层中，会将这个嵌入矩阵~~(共享权重矩阵)~~乘以$\sqrt{d_{model}}$。(**修正**)

共享权重取决于具体的任务需求和设计选择：

- 如果源语言和目标语言的词汇表是相同的（例如在某些双语翻译任务中；或至少要有较严苛的一一对应关系），或者两者的嵌入在相同的语义空间中有意义，那么共享权重可以减少参数数量，并可能提高模型的泛化能力。
- 在多数情况下，源语言和目标语言的词汇表是不同的，因此通常会使用独立的嵌入层。这可以让模型更灵活地适应不同的语言特性。



#### 掩码机制

**对原(输入)序列**： 输入序列中包含的填充字符`<pad>`对句子信息无意义，因此无需计算其其他词对它的注意力。所以对输入序列要找到它的位置创建`src_mask`。(src means source)

**对目标序列**：由于Transformer具有并行计算和并行输出的特点，而每个位置的输出应该只依赖于其以及其前面位置的信息(自注意力的计算)，故目标序列计算自注意力时需添加掩码，以遮盖未来位置的词。即任意自注意力权重$QK^\top=\{q_ik_j\}$ 中满足$i>j$的位置应用`tgt_mask`标记。(tgt means target)

论文的做法是在softmax之间将这些值设为负无穷，其对应分数自然变为0(不再关注)。

**目标序列掩码矩阵的实现(之一)**：

1. 生成掩码矩阵：$M=\begin{bmatrix}0&-\infin&\cdots&-\infin\\0&0&\cdots&-\infin\\ \vdots&\vdots&\ddots&\vdots\\0&0&\cdots&0 \end{bmatrix}\in \mathbb R^{M\times M} $

2. 解码器首个自注意力层计算：$Z=\text{softmax}(\frac{QK^\top+M}{\sqrt {d_k}})\cdot V$，使得任意注意力权重失效。

#### 正则化

为了提高 Transformer 模型的性能，在训练过程中，使用了以下的正则化操作：

1. Dropout。对编码器和解码器的每个子层的输出使用 Dropout 操作，是在进行残差连接和层归一化之前。词嵌入向量和位置编码向量执行相加操作后，执行 Dropout 操作。Transformer 论文中提供的参数$ P_{drop} = 0.1$

2. Label Smoothing（标签平滑）。Transformer 论文中提供的参数$\epsilon_{ls} = 0.1$

在 Transformer 模型中，正则化主要通过以下方式实现：

1. **Layer Normalization**：(已有)

   - `LayerNorm` 在每个编码器和解码器层中用于标准化特征维度，帮助网络更稳定地训练。
   - 在 `EncoderLayer` 和 `DecoderLayer` 中，LayerNorm 用于处理自注意力输出和前馈网络的输出。

2. **Dropout**：(额外)

   - Dropout 是一种随机关闭部分神经元的方法，用于减少过拟合并提高模型的泛化能力。
   - 在 Transformer 中，Dropout 被用于：
     - 输入嵌入和位置编码的组合。
     - 自注意力和前馈网络的输出。

3. **Residual Connections**：(已有)

   - Transformer 使用了残差连接（`x + ...`），通过跳跃连接，保留了输入特征，同时缓解梯度消失问题。
   - 每个子层（自注意力、前馈网络）的输出都会通过残差连接加回原输入。

4. **Label Smoothing**：(额外)

   - 标签平滑是一种正则化技术，用于提高模型的泛化能力并防止过拟合，特别是在分类任务中。

   - 假设有C个类别，目标类别的分布是$q = [0, 0, ..., 1, ..., 0]$，应用标签平滑后，目标分布变为：

     $q' = (1 - \epsilon) \cdot q + \epsilon / C$

     其中：

     - $\epsilon$ 是平滑系数（例如 0.1）。
     - $1 - \epsilon$ 是正确类别的权重。
     - $\epsilon / C$ 是错误类别的权重（均匀分布）。

#### 其它

- 两个嵌入层后都乘了系数$\sqrt{d_{model}}$



## 算法实现-Pytorch

一份使用`colab`的实现，方便任何时候输出查看`tensor`维度：<https://colab.research.google.com/drive/1lbc9RoMMgUH6hCBccmjxPaHS_Q3TFvja?usp=sharing>

### 数据集准备

数据集的选取参考[4]

**说明:** 

> 该数据集由`cn.txt`, `en.txt`, `cn.txt.vocab.tsv`, `en.txt.vocab.tsv`这四个文件组成。前两个文件包含相互对应的中英文句子，其中中文已做好分词，英文全为小写且标点已被分割好。后两个文件是预处理好的词表，包含频次统计。语料来自2000年左右的中国新闻。
>
> 词表的前四个单词是特殊字符，分别为填充字符`<pad>`、频率太少没有被加入词典的词语`<UNK>`、句子开始字符`<s>`、句子结束字符`</s>`。

1. 读取的原始序列：
   - `src_seq`: `['the','present',...,'landscapes','.']`
   - `tgt_seq`: `['目前','粮食',...,'山川','。']`

2. 根据每个词在词表的位置`idx`编码成向量：
   - `src_seq`: `[32, 89, 10, ..., 77]`
   - `tgt_seq`: `[41, 8, 0, ..., 99]`
3. 添加开始`<s>`与结束符`</s>`，并用`<pad>`补全至最大`max_len`
   - `src_seq`: `[1, (32, 89, 10, ..., 77), 2, 0,..., 0]`
   - `tgt_seq`: `[1, (41, 8, 0, ..., 99), 2, 0,..., 0]`
4. `Dataloader`封装

### 模型搭建

Transformer 的最后一层通常 不直接进行 Softmax 操作，而是在计算损失时由 `CrossEntropyLoss` 内部完成 Softmax 的操作。这种设计具有以下原因和优点：

1. **Softmax 和 CrossEntropy 的计算效率**

- `CrossEntropyLoss` 整合了 Softmax 和负对数似然（Negative Log-Likelihood, NLL）的计算。
- 将这两步合并在损失函数内部可以减少数值不稳定性问题，例如溢出或精度损失。
- 因此，模型输出直接是 logits，避免重复计算 Softmax，从而提高效率。

2. **输出维度与损失计算的匹配**

- 模型最后输出的维度为 `(batch_size, tgt_seq_len - 1, tgt_vocab_size)`，其中 `tgt_vocab_size` 是目标词表的大小，表示每个时间步上对各词的原始分数（logits）。
- `CrossEntropyLoss` 会自动将 logits 转换为概率分布（通过 Softmax）并与目标 `tgt_output` 比较。

3. **灵活性**

- 不在模型中强制加 Softmax，可以更灵活地应用于不同任务。例如在推理阶段可能不需要 Softmax，而是使用 `argmax` 或其他操作来选择最高分数的标记。

### 训练

训练逻辑：(每个epoch)

1. 从`Dataloader`中抽取`batch_size`
   - `src`: `(b, max_len)`
   - `tgt`: `(b, max_len)`
2. `src`经过嵌入层进入Encoder->`(b, N, d_model)`
3. `tgt`去掉最后一个token后`tgt_input`作为Encoder的输入 `(batch_size, tgt_seq_len - 1)`
4. `tgt`去掉第一个token后`tgt_output`作为目标输出(即标签)
5. 计算Transformer的输出`output`与`tgt_output`的`CrossEntropyLoss`作为损失

事实上，`tgt_output`正是下一个时间步模型的期望输出，或反过来说，`tgt_input`是理想的上一步的预测，为`Decoder`提供上下文信息。为了保持并行计算并行输出的优势，计算自注意力时使用`mask`遮住所在位置后面来自未来预测的信息。



### 推理

待翻译序列：`[<s>, we, should, protect, our, environments, ., </s>, <pad>, ..., <pad>]`

`for _ in range(max_len):`

每次循环Encoder的输入->输出:

1.  `[<s>]->[我们]` 取输出的最后一个token与输入拼接组成下一次的输入
2.  `[<s>, 我们] -> [我们, 应该]`
3.  `[<s>, 我们, 应该]->[我们, 应该, 保护]`
4.  `[<s>, 我们, 应该, 保护]->[我们, 应该, 保护, 我们]`
5.  `[<s>, 我们, 应该, 保护, 我们]->[我们, 应该, 保护, 我们, 的]`
6.  `[<s>, 我们, 应该, 保护, 我们, 的]->[我们, 应该, 保护, 我们, 的, 环境]`
7.  `[<s>, 我们, 应该, 保护, 我们, 的]->[我们, 应该, 保护, 我们, 的, 环境, </s>]` 结束符检测，break

**补充**：

最开始测试推理用来翻译的句子是：`"we should protect environment."`，结果总是很怪，比如：

```bash
Epoch 50, Loss: 0.0829
===================
50 :  对 我们 应该 保护 <UNK> 粮食 的 同时 <UNK> 要 加大 力度 <UNK> <UNK> 保护 <UNK> 自己 <UNK>

Epoch 60, Loss: 0.0597
===================
60 :  在 应该 是 粮食 呢 <UNK> 我们 应当 要 善于 应当 善于 <UNK> 粮食 <UNK> <UNK> 我们 应当 自己 是 增加 <UNK>

Epoch 70, Loss: 0.0453
===================
70 :  作为 粮食 的 地理 应该 是 粮食 播种 <UNK> 我们 应当 包括 自己 的 劳动力 <UNK> 应 善于 <UNK> 自己 <UNK> <UNK>

Epoch 80, Loss: 0.0353
===================
80 :  我们 应该 要 保护 <UNK> 在 <UNK> 三 讲 <UNK> <UNK> 保护 <UNK> 我们 应当 将 自己 的 应 成为 <UNK>

Epoch 90, Loss: 0.0311
===================
90 :  在 我 实施 起来 <UNK> 我们 应当 包括 <UNK> 亿 <UNK> 我们 应当 写 进 <UNK> 提高 上述 <UNK> 吗 <UNK>

Epoch 10/10, Loss: 0.0268
===================
100 :  <UNK> 我们 应当 <UNK> 十五 <UNK> <UNK> 应 安排 的 要 运用 <UNK> 我们 应当 善于 游戏 规则 <UNK>
```

检查了一下发现`sentence`后面的英文句号应该和训练数据集一样用空格分隔开来 (数据预处理)

之后结果就好很多了

```json
Source Sentence: we should protect our environments .
Translated Sentence: 我们 要 保护 的 保卫 <UNK> 节约 应该 把 工作 <UNK> 保护 网络 <UNK> 保卫 培养 等 <UNK>
```

然后我把句子换成和参考[4]一样的句子想看看效果: `"we should protect environment ."`

结果是：

```json
Source Sentence: we should protect environment .
Translated Sentence: 要 保护 环境 保护 环境 <UNK> 抓住 环境 <UNK> 我们 应当 切实 要 保护 环境 <UNK> 保护 环境 <UNK> 环境 <UNK> 保护 环境 <UNK>
```

和[4]一样很奇怪，为什么`保护`和`环境`之后不会终止呢？([4]的终止检测应该有问题，翻译里居然出现了多个终止符)

然后我突然想到句号在词库里是怎样表达的，果然一查没有这个`token`，或者说归纳到`<pad>`或者`<UNK>`里了，显然作为重要的句子结束标记(个人认为)这样的词库不合理。。。

所以我去掉了翻译句子里的`.`试了一下：

```json
Source Sentence: we should protect environment
Translated Sentence: 我们 应当 要 抓好 的 环境 <UNK> 保护 好 环境 <UNK> 使 我们 应当 成为 环境 的 环境 <UNK>
```

虽然依旧结束位置很奇怪 (词库本身问题)，但基本对应的词都算是有了。**Win！**

### 总结

> ​	总体来说，Transformer架构下，其编码组件可视作特征提取模型，而解码组件可视作生成模型，而注意力机制的提出具有开创性，也可很大的拓展性和可能性。

## 参考

- [[0] Attention is All you need](https://arxiv.org/abs/1706.03762)
- [[1] 经典的Transformer图解英文博客](https://jalammar.github.io/illustrated-transformer/)
- [[2] 图解的中文翻译](https://blog.csdn.net/benzhujie1245com/article/details/117173090)
- [[3] 一篇不错的理论博客(一)-ZhouYifan](https://zhouyifan.net/2022/11/12/20220925-Transformer/)
- [[4] 一篇不错的实现博客(二)-ZhouYifan](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)
- ChatGPT: 强推，胜过任何老师
