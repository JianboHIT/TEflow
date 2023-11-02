==============
载流子输运模型
==============

材料的电输运性质与材料的能带结构通常密切相关。
在实际中我们通常也是花费很多的精力来进行材料电子结构的设计，从而优化材料的热电性质。
对于对于热电半导体材料而言，
其热电性质通常都是当费米能级位于导带底或者价带顶时获得一个峰值的功率因子与热电优值。
因此，在实践中发展出了一系列的热电输运模型以及衍生结论。
我们已经尽量最大努力来构建一个通用的计算框架来实现多种研究目的，
比如估算有效质量，权重迁移率，或者品质因子，再或者是为了探究一个假想材料的性能极限，
比如估计一个双能带体系当其能带劈裂值完全降低到 0 时的性能上限从而评估研究的必要性。

基本原理
--------

载流输运的核心在于玻尔兹曼输运方程，所有的热电性质都与材料的输运分布函数
(或者谱电导率)，定义为：

.. math::
    :label: bte

    \sigma_s(E,T) = q^2 \tau v_{xx}^2 g(E)

材料的电导率可以认为是费米能级附近载流子的谱电导率的平均值。准确地说，
是谱电导率和费米窗口函数的卷积，

.. math::
    :label: sigma

    \sigma(E_F, T) = \int_{-\infty}^{+\infty} \sigma_s(E, T)
        w_F(E; E_F, T) dE

其中，费米窗口函数定义为：

.. math::
    :label: wfermi

    w_F(E; E_F, T) 
        &= -\frac{\partial f(E; E_F, T)}{\partial E} \\
        &= -\frac{\partial}{\partial E} \left[
                \frac{1}{1+\exp \left( \cfrac{E-E_F}{k_B T} \right)}
            \right] \\
        &= \frac{1}{k_B T}
           \frac{\exp \left( \cfrac{E-E_F}{k_B T} \right)}{
                \left[1+\exp \left( \cfrac{E-E_F}{k_B T} \right)
                \right] ^2
           }
        
基于玻尔兹曼输运理论，我们可以进一步给出塞贝克系数和电子热导率：

.. math:: 
    :label: seebeck

    S = \frac{k_B}{\sigma q}
        \int_{-\infty}^{+\infty} \sigma_s(E, T)
        \frac{E-E_F}{k_B T}
        w_F(E; E_F, T) dE

.. math:: 
    :label: kappa

    \kappa_e = T \left( \frac{k_B}{q} \right) ^2
        \int_{-\infty}^{+\infty} \sigma_s(E, T)
        \left(\frac{E-E_F}{k_B T}\right) ^2
        w_F(E; E_F, T) dE
        -T \sigma S^2

程序实现
--------

当前相关计算程序已经完成
(详见 :doc:`bandlib </api_doc/teflow.bandlib>` 模块)，
接口还在开发中。
