---
layout: post
title: "Description of Connection Subgraph Resources in BCE Architecture"
date: 2024-08-13 17:00:00 +0800
categories:
  - Report 
tags:
  - 进度汇报
---
我这两天在研究簇定位算法，简单来说，我这个算法的目的是在考虑尽可能地减少不同NFU（Neural Functional Unit）上的簇进行通讯的距离，将彼此通信次数多的尽可能较近排列；以及在每个NFU内部，对簇之间也进行类似操作。

Notice：昨天提到，不同的神经元之间存在网络拓扑层级，那么不同的簇之间，也应该是有层级的。

我之前的想法没有考虑到这个因素，对于这个特性，我还没想到用什么方式调优。因此，我目前只能分享一下我目前的思路，以及算法中应该会用到的一些基础算法。

Intuitive Algorithm:

Step 0 数据接入。

Step 0.5
根据GNC的层级，新赋一个参数L，来作为一个奖励分数，鼓励层级相近的GNC放置在一起。

Step 1
用模拟退火算法将GNC初略地划分到不同地NFU，尝试找出全局最优。

Step 2
对于粗划分的NFU表，用Kernighan-Lin minimum cut算法进行再次划分，尝试局部微调。

Step 3
对于每个NFU内部的GNCs，应用HSC等空间填充曲线进行填充。

Step 4
对填充后的NFU，应用FD或模拟退火算法，使连接较多的GNC更接近。

相关算法：

1, Kernighan-Lin minimum cut
这个算法主要是对按某种策略将一个集合中的节点分成两个子集所构成的割集最小化（即减小两个子集连接边的数量/权重大小）

基本思路是：假设有两个子集，分别为A, B.
那么，对于任意节点a∈A, 有评估指标D(a):

D(a) = (a到B中所有节点的cost的和) - (a到A中所有除自身以外所有节点的cost 的和), D(b)同理

之后， 找到A,B子集中所有没有交换过的节点，遍历计算g(i) = D(a) + D(b) - 2c(a, b)的值。将其中使得g(i)最大的(a, b)对进行交换。
```Pseudocode
    1. i = 0
    2. 当两个子集中仍然能找到没有交换的对:
        (a) i = i + 1 
        (b)	Find unexchanged a∈A and b∈B such that gi = D(a) + D(b)- 2c(a, b) is maximized.
        (c)	Exchange the pair consisting of a and b.
        (d)	Recalculate the D-values for the vertices that have not been exchanged before.
    3.	Find k such that G = g1 + • • • + gk is maximized.
    4.	Switch the first k pairs
重复上述操作，直到 G = 0 或者达到指定循环次数。
```
2，没有做完