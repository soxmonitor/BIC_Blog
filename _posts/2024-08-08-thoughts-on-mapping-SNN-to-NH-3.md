---
layout: post
title: "Thoughts on Mapping SNN to NH(3)"
date: 2024-08-08 14:45:00 +0800
categories:
  - Views on essay
tags:
  - thoughts
  - snn
  - reason
  - partitioning
  - clustering
---
为什么<b>簇排序</b>(clustering)在<b>映射</b>(mapping)过程中如此重要呢？

神经元通常只与附近的少数其他神经元连接，而不是在整个网络中广泛连接。因此，找到好的簇的空间排列方式有助于将具有连接关系的神经元映射到二维空间中的相近位置，以减少长度和延迟。

目前，对于这一目标，有一种观点是采用某种空间填充曲线，原因在于：

<br>
1.<b>局部性</b>：部分空间填充曲线，如Hilbert Space Filling Curve(HSC), 具有较为优秀的<b>局部性</b>(locality)，即在一维空间中彼此靠近的两个点在映射到二维空间后也会彼此靠近。不同的空间填充曲线之间的局部性之间存在较大的差异。

希尔伯特曲线演示：

<iframe src="/assets/blog2/hilbert.html" width="420" height="420" style="border:none;"></iframe>

希尔伯特3D曲线演示：
[3D Hilbert Curves演示](https://observablehq.com/@mourner/3d-hilbert-curves)。


下面四张图分别是HSC, ZigZag Curve, Circle(Onion) Curve 及 Z-order Curve 将 y=0 这条直线进行翻折以填充一个8x8的二维空间时，生成的新点(x<sup>'</sup>,y<sup>'</sup>)到原始点（0，y）的距离构成的热力图。距离原始点越近，则颜色越深。

<img src="/assets/blog3/HSC.jpg" alt="HSCon8x8">
<img src="/assets/blog3/Zigzag.jpg" alt="Zigzagon8x8">
<img src="/assets/blog3/Circle.jpg" alt="Circleon8x8">
<img src="/assets/blog3/Z-order.jpg" alt="Z-orderon8x8">

根据上面的图像，生成了四幅新图。在每一幅图中，(x,y)的值为从对应前一张图中随机抽取的Px, Py 两点之间的距离，并根据此生成的热力图
<img src="/assets/blog3/HSCon8x8_Prob.png" alt="HSCon8x8_Prob">
<img src="/assets/blog3/Zigzagon8x8_Prob.png" alt="Zigzagon8x8_Prob">
<img src="/assets/blog3/Circleon8x8_Prob.png" alt="Circleon8x8_Prob">
<img src="/assets/blog3/Z-orderon8x8_Prob.png" alt="Z-orderon8x8_Prob">

对于一个未知的SNN，我们生成一个概率云图，该云图为许多目前已知SNNs训练而成。将云图作为一个掩码(mask)与上面四幅新图相加，得到Cost。比较Cost即可得出最佳的空间填充曲线。

<img src="/assets/blog3/PC.png" alt="Probably Cloud">

经过计算后，可以得到目前HSC其它空间填充曲线得到的Cost低，因而采用HSC。

该比较方法出自[Mapping  very  large  scale spiking  neuron  network  to  neuromorphic  hardware](/assets/blog3/Mapping_very_large_scale_spiking_neuron_network_to_neuromorphic_hardware2023.pdf)

下文是从数学方面比较HSC和另一常见空间曲线Z-Order Curve特性的论文，并论证了HSC的局部性比z-Order更佳。

[Clustering Analyses of Two-Dimensional Space-Filling Curves: Hilbert and z-Order Curves](/assets/blog3/Clustering_Analyses_of_Two-Dimensional_Space-Filling.pdf)

<br>

<b>需要注意的是</b>: Locality的优劣并不能单独决定一个空间填充的好坏,事实上，[Circle(Onion) Curve的Locality更加优秀](/assets/blog3/Onion_Curve.pdf)，然而在上面的评估中，由于概率云的叠加，Circle Curve表现比HSC差。但是反过来说，<b>有可能存在Locality不如HSC的空间填充曲线，在评估中表现却更佳。</b>


<br>

2.<b>无穷性</b>：空间填充曲线可以通过无限迭代填充一个二维空间。也就是可以提供任意大小的二维空间与一维空间之间的映射，这意味着在将非常大规模的SNN应用映射到硬件时不会出现扩展性问题​。

下文是一篇关于如何将HCS应用到非2<sup>n</sup>矩形空间的论文

[Modified Hilbert Curve for Rectangles and Cuboids.pdf](/assets/blog3/Modified_Hilbert_Curve_for_Rectangles_and_Cuboids.pdf)

<br>
3.<b>方向性</b>: 在经过拓扑排序后的簇序列中，数据流从输入端到输出端像水流一样具有方向性。而大部分空间填充曲线在映射过程中保持了这种方向性。并且，大部分空间填充曲线是分形图，这意味着它们在任何它们的子图中都具有这种特性。因此，不仅在全局范围内，而且在任何小局部区域，这种特性都有助于获得更好的布局。
