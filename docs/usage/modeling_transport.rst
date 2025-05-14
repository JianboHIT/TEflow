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

波尔兹曼输运理论
^^^^^^^^^^^^^^^^

载流输运的核心在于玻尔兹曼输运方程，所有的热电性质都与材料的输运分布函数
(或者称为谱电导率)，定义为：

.. math::
    :label: bte

    \sigma_s(E,T) = q^2 \tau v_{xx}^2 g(E)

材料的电导率可以认为是费米能级附近载流子的谱电导率的平均值。准确地说，
是谱电导率 :math:`\sigma_s(E, T)` 和费米窗口函数 :math:`w_F(E; E_F, T)` 的卷积，

.. math::
    :label: sigma

    \sigma(E_F, T) = \int_{-\infty}^{+\infty} \sigma_s(E, T)
        w_F(E; E_F, T) dE

其中，费米窗口函数 :math:`w_F(E; E_F, T)` 以及费米狄拉克分布函数
:math:`f(E; E_F, T)` 定义为：

.. math::
    :label: wfermi

    w_F(E; E_F, T) 
        = -\frac{\partial f(E; E_F, T)}{\partial E}
        \equiv \frac{1}{k_B T} \cdot f(E; E_F, T) \cdot [1-f(E; E_F, T)]

.. math::
    :label: fermidirac

    f(E; E_F, T) = \cfrac{1}{1+\exp \left[ (E-E_F)/(k_B T) \right]}

费米窗口函数 :math:`w_F(E; E_F, T)` 是一个类似高斯分布的单峰值函数，如下图所示：

.. plot::
    :align: center
    :height: 300px

    xdata = np.linspace(-10, 10, 201)
    ydata = 1/4 * (1-np.power(np.tanh(xdata/2), 2))
    plt.plot(xdata, ydata, label='Scaled $w_F(E; E_F, T)$')
    plt.xlabel('$(E-E_F)/(k_B T)$')
    plt.ylabel('$w_F(E; E_F, T) \\times k_B T$')
    plt.xlim(xdata.min(), xdata.max())
    plt.ylim(0, 0.3)
    plt.legend()

基于玻尔兹曼输运理论，我们可以进一步给出塞贝克系数和电子热导率：

.. math:: 
    :label: seebeck

    S = \frac{k_B}{\sigma q}
        \int_{-\infty}^{+\infty} \sigma_s(E, T)
        \frac{E-E_F}{k_B T}
        w_F(E; E_F, T) dE

.. math:: 
    :label: kappa_e

    \kappa_e = T \left( \frac{k_B}{q} \right) ^2
        \int_{-\infty}^{+\infty} \sigma_s(E, T)
        \left(\frac{E-E_F}{k_B T}\right) ^2
        w_F(E; E_F, T) dE
        -T \sigma S^2

并且可以按照定义导出洛伦兹常数:

.. math:: 
    :label: lorenz

    L = \frac{\kappa_e}{\sigma T}

由此，我们看到载流子的输运问题核心在于谱电导率 :math:`\sigma_s` 
(即公式 :eq:`bte`)的处理，有了它我们就能够定量计算其它的几个热电输运参数。
接下来，我们将从这个角度来理解经典的热电输运模型。

单抛物带模型
^^^^^^^^^^^^

对于半导体材料而言，费米能级通常认为位于能带的边缘，
即价带顶(空穴)或者导带底(电子)。在这样一个极值点附近，
我们通常可以用一个抛物线来很好地描述能带形状，即

.. math::
    :label: para_band

    E = \frac{\hbar ^2 k ^2}{2 m _b ^\ast}

这里 :math:`k` 是波矢量，:math:`m_b ^\ast`
是能带有效质量。参照公式 :eq:`bte`, 
我们注意到当色散关系确定后，就已经可以得到态密度和群速度
(暂时先假定能带各向同性):

.. math:: 
    :label: para_dos

    g(E) = \frac{\left(2 m _b ^\ast \right) ^{\frac{3}{2}}}
        {2 \pi ^2 \hbar^3} \sqrt E

.. math:: 
    :label: para_vg

    v _g = \frac{1}{\hbar} \frac{\partial E}{\partial k}
        = \frac{\hbar k}{m _b ^\ast}

.. math::
    :label: para_vx

    v_{xx}^2 = v_{yy}^2 = v_{zz}^2
        = \frac{1}{3} v_g^2
        = \frac{1}{3} \left( \frac{\hbar k}{m _b ^\ast} \right)^2
        = \frac{2}{3} \frac{E}{m _b ^\ast}

至此，我们的核心就落到了载流子的迟豫时间 :math:`\tau` 上。
对于实际的材料而言，载流子的迟豫时间会受到各种散射机制的影响，
比如声学声子的散射，电离杂质的散射，合金化元素的散射，
甚至还有载流子之间的相互散射等等。
一般我们定义载流子迟豫时间的倒数为散射概率，
总的散射概率为各种散射机制的概率加和，
表现在迟豫时间上即为倒数加和规律:

.. math::
    :label: tau_e_tot

    \frac{1}{\tau _{tot}} = 
        \frac{1}{\tau _{ph}}
        + \frac{1}{\tau _{imp}}
        + \frac{1}{\tau _{alloy}}
        + \frac{1}{\tau _{ele}}
        + \cdots

长波声子散射
^^^^^^^^^^^^

在众多的载流子散射机制中，
长波声学声子散射(APS)是最为基本的散射机制，
描述了晶格振动对于电子输运的影响。
而且，其它的散射机制强度会严重地依赖材料种类，温度，
或者掺杂强度等因素，只有 APS 机制会始终伴随材料，
尤其在高温时，由于声子的大量激发(即晶格振动剧烈),
APS 通常会完全占据主导地位。
可以想见，由于 APS 机制本质就是电子与声子的碰撞，
它的强度会正比于电子态密度，即

.. math:: 
    :label: tau_dos

    \tau _{APS} ^{-1} = \frac{1}{\lambda} \cdot g(E)
        \text{, i.e. } \tau _{APS} \cdot g(E) = \lambda

在形变势理论框架下，可以推导出

.. math:: 
    :label: tau_dp

    \lambda = \frac{\hbar C_{ii} N_v}{\pi k_B T E_d^2}

这里, :math:`C_{ii}` 是弹性常数， :math:`N_v` 是能带简并度，
:math:`E_d` 是形变势。
至此，我们就能够完全定量确定材料的谱电导率 :math:`\sigma_s`,
利用方程 :eq:`sigma` ~ :eq:`kappa_e` 就能够确定载流子的输运参数。

载流子工程
^^^^^^^^^^

从前面载流子的输运讨论可以看到，当能带和散射机制完全确定时，
热电输运参数直接依赖于温度和费米能级。因此，
为了获得更好的热电性能，掺杂调控是一种最为基本的调控策略，
通过影响费米能级位置直接影响着材料的性能。
从另外一方面来说，这也是热电材料研究的重要阻碍之一，
即在进行完全的掺杂优化之前，我们很难直接断言材料性能的优劣。
因此，我们需要借助一些理论的探究，
来帮我们更加准确高效地鉴别材料最终获得高性能的潜力，
以及辅助我们进行掺杂浓度的调控和设计。

声学声子散射主导的单抛物带模型(APS-SPB),
是目前为止应用最为广泛和成功的载流子输运模型。
在这样一个模型框架下，我们可以将热电相关性质写成

.. math:: 
    :label: spb_eta

    & \sigma = \sigma _0 F_0(\eta) \\
    & S = \frac{k_B}{q}
        \left( \frac{2F_1(\eta)}{F_0(\eta)} - \eta \right) \\
    & L = \left( \frac{k_B}{q} \right) ^2 \left[ 
        \frac{3F_2(\eta)}{F_0(\eta)}
        - \left( \frac{2F_1(\eta)}{F_0(\eta)} \right) ^2 
        \right]

其中，引入了约化费米能级 :math:`\eta` 和费米积分 :math:`F_n(\eta)`,
分别定义为:

.. math::
    :label: df_eta

    \eta = \frac{E_F}{k_B T}

.. math:: 
    :label: df_fn

    F_n(\eta) = \int _0 ^{+\infty} \frac{x^n}{1+\exp(x-\eta)}dx

这里，我们只需要将费米积分简单地理解成一簇特殊数学函数即可。
为了更加直观地它，我们在下面给出了它在线性刻度和对数刻度下的图像。

.. plot::
    :align: center
    :height: 300px

    from teflow.mathext import vquad, fermidirac

    Fn = lambda x, eta, n: np.power(x, n) * fermidirac(x-eta)
    x_eta = np.arange(-5, 20, 0.1)
    n = np.arange(3)
    y_Fn = vquad(Fn, 0, np.inf, args=(x_eta[..., None], n))[0]
    width = plt.rcParams['figure.figsize'][0] * 1.75
    height = plt.rcParams['figure.figsize'][1]
    plt.figure(figsize=(width, height))
    plt.subplot(121)
    plt.plot(x_eta, y_Fn, label=[f'n = {i}' for i in n])
    plt.xlabel('$\eta$ = $E_F/(k_B T)$')
    plt.ylabel('$F_n(\eta)$')
    plt.legend()
    plt.subplot(122)
    plt.plot(x_eta, y_Fn, label=[f'n = {i}' for i in n])
    plt.xlabel('$\eta$ = $E_F/(k_B T)$')
    plt.ylabel('$F_n(\eta)$')
    plt.yscale('log')


在经典的相关讨论中，通常会进一步引入简并近似(适用 :math:`\eta \ll 0`)
和非简并近似(适用 :math:`\eta \ll 0`)来简化费米积分 :eq:`df_fn`,
从而简化前面的热电系数 :eq:`spb_eta`。
但是由于实际的功率因子峰值和热电优值的峰值都通常出现在
:math:`\eta \approx 0` 时，因此我们这里不再讨论相关内容，
而是放在讨论这些表达式的特征分析上。

首先，我们注意到所有的热电输运系数都直接依赖于约化费米能级 :math:`\eta` ,
而且塞贝克系数和洛伦兹常数仅依赖 :math:`\eta` 。
基于这一点，我们可以通过试验塞贝克系数可以求解出 :math:`\eta` ,
进一步求解出洛伦兹常数。
除了 :math:`\eta` 以外, 输运系数就仅仅依赖一个半经验的参数:
本征电导率 :math:`\sigma_0` 。
考虑材料的功率因子，

.. math::
    :label: spb_pf
    
    PF = \sigma S^2 
        = \sigma_0 \cdot \left( \frac{k_B}{q} \right) ^2
        \frac{\left[ 2F_1(0) - \eta F_0(\eta)\right] ^2}{F_0(\eta)}
        = \sigma_0 \left( \frac{k_B}{q} \right) ^2 \cdot PF_r(\eta)

我们注意到，功率因子可以被拆分成为一个比例系数 :math:`\sigma_0`
和一个仅依赖 :math:`\eta` 的和材料无关的特殊数学函数 :math:`PF_r(\eta)` 。
这里，:math:`PF(\eta)` 是一个典型的单峰值形状的函数，
在 :math:`\eta \approx 0` 时取得最大值，约为 4.02。
因此，对于一个材料而言，:math:`\sigma_0` 将是决定其最大
:math:`PF` 的唯一因素，也是我们进行能带模型的核心。
在实际中，它被演化成其它一些我们熟悉的概念，
比如权重迁移率，电学品质因子等等，
本质上它们是完全等价的，只是相差一个常数的缩放因子。

Kane能带模型
^^^^^^^^^^^^

对于一个实际的热电材料而言，
为了获得高的热电性能我们必须要进行重掺杂。
此时，费米能级将会进入导带或者价带内部，
为了更好地评估材料的热电性能，
我们必需要考虑能带偏离理论抛物线时的影响。
从直觉上我们可能会引入更高阶次的多项式来描述能带，
提升我们对于热电性质评估的准确性，
但是这并没有给我们带来更加清晰的物理含义。
在实践中我们发现, Kane 模型是一个比较恰当的模型,即

.. math:: 
    :label: kane_band

    E \left( 1+\frac{E}{E _\Delta} \right) = 
        \frac{\hbar ^2 k ^2}{2 m _b ^\ast}

对于经典的抛物带模型 :eq:`para_band` , 
我不难发现当 :math:`E _\Delta \rightarrow +\infty`
时, Kane 模型将退化成为抛物带模型。
对于实际的材料，当我们尝试用 Kane 模型去拟合，
通常都会对应一个有限的 :math:`E _\Delta` 值。
从数学角度来看，这是一个偏移的双曲线模型，
随着能量，它会逐渐靠近其渐近线，呈现出线性的色散关系。
在这样的色散关系下，我们也可以给出相关热电性质的描述。

程序实现
--------

基于能带的输运模型在实际中有很多应用，
比如通过塞贝克系数求解洛伦兹常数，通过 Pisarenko 关系计算有效质量，
基于实验结果估计材料的权重迁移率并估计最大功率因子，等等。
目前，我们提供了一个比较初级的命令行接口 ``tef-band`` ,
实现比较常规的试验数据分析，
更加复杂的功能需要通过调用相应的模块来完成。

命令行指令
^^^^^^^^^^

我们可以通过 ``tef-band -h`` 选项来查看帮助，

.. code-block::

    $ tef-band -h
    usage: tef-band [-h] [-H] [-b] [--T VALUES] [--EF VALUES] [--deltas DELTAS]
                    [--btypes BTYPES] [--initial INITIAL] [-m {SPB,RSPB,SKB}]
                    [-G GAP] [-g GROUP] [-p PROPERTIES] [-s SUFFIX]
                    INPUTFILE [OUTPUTFILE]
    
    Insight carriar transport with band models - TEflow(0.2.7a1)
    
      >>> Constructs a rigid multi-band model from band parameters (defined in
      a configuration file) to simulate thermoelectric performance of materials
      across any Fermi-level and temperatures. Additionally, a popular feature,
      activated by the -m/--modelling option, allows for the rapid evaluation
      of experimental data through either the classical parabolic band or the
      Kane band model.
    
    positional arguments:
      INPUTFILE             Input file name (must be provided)
      OUTPUTFILE            Output file name (optional, auto-generated if omitted)
    
    optional arguments:
      -h, --help            show this help message and exit
      -H, --headers         Include headers without a hash character
      -b, --bare            Output data without header
      --T VALUES            Override 'T' value in entry section
      --EF VALUES           Override 'EF' value in entry section
      --deltas DELTAS       Override 'deltas' value in entry section
      --btypes BTYPES       Override 'btypes' value in entry section
      --initial INITIAL     Override 'initial' value in entry section
      -m {SPB,RSPB,SKB}, --modelling {SPB,RSPB,SKB}
                            Directly insight experimental data using the selected model.
      -G GAP, --gap GAP     Bandgap in eV, required by SKB model.
      -g GROUP, --group GROUP
                            Group identifiers for input data (default: STCN)
      -p PROPERTIES, --properties PROPERTIES
                            Specify the properties to be considered for calculation, separated by spaces.
      -s SUFFIX, --suffix SUFFIX
                            Suffix for generating the output file name (default: band)

这里可以通过 ``-m/--modelling`` 选项来选择数据分析的模型，
输入文件需要包含塞贝克系数，可选地还可以包含温度，
电导率和载流子浓度。根据所给的输入数据，
输出数据可以包括：洛伦兹常数，温度无关权重迁移率，有效质量等。
对于 Kane 模型，还需要通过 ``--gap <Egap>`` 选项来指定带隙。
假设我们材料的带隙为 0.1 eV, 输入文件名为 data.txt,
我们应该像下面这样操作:

.. code-block:: bash

    $ tef-band -m SKB --gap 0.1 data.txt

相关的模块
^^^^^^^^^^

我们提供了比较完整的函数和类文档，具体可以参考
:doc:`bandlib </api_doc/teflow.bandlib>`
模块。

.. code-block:: python3

    >>> from teflow.bandlib import APSSPB
    >>> 
    >>> # construct SPB model by classmethod from_DP()
    >>> spb = APSSPB.from_DP(m1=1, Ed=10, Nv=2, Cii=10)
    >>> spb.S(EF=-0.01, T=300)
    229.2960316083183
