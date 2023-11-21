============
工程热电优值
============

热电器件的输出功率和转化效率是我们非常关注的两个指标。
对于经典的 :math:`\Pi` 型器件，
如果考虑材料的性能不随温度变化，
我们是可以导出其输出功率和转化效率与材料性质的解析关系的，
还由此可以得到热电材料的两个性能指标: 功率因子和无量纲热电优值。
当考虑到材料性能随温度的复杂变化关系后，
我们无法再给出一个显式的解析解，只能考虑数值解法。
其中一个代表性的工作就是 Snyder 等人在定义器件优值时，
同时附带给出了一个可以进行差分求解器件效率的 Excel 工具。
但是，对于成对器件，这种方式就比较乏力了，
不但效率较低，而且不易处理器件尺寸的优化协调问题。
在此背景之下，刘等人提出了工程热电的相关理论，
在充分考虑材料性能温度依赖性的前提下，
给出了器件性能的半解析公式，以及一个修正后的功率因子和热电优值。
我们的程序包已经实现了单腿和双腿工程热电性能计算的功能，
并且提供了命令行接口。

基本原理
--------

热电器件性能评估的本质是在求解一个电-热深耦合物理场的问题，
其核心问题可以归结为求解下面一个二阶微分方程:

.. math::
    :label: governing_eq

    \frac{d}{dx}\left( \kappa \frac{dT}{dx} \right)
        + J^2 \rho - J \tau \frac{dT}{dx} = 0

这里 :math:`\kappa` 、:math:`\rho` 和 :math:`\tau`
分别是热导率、电阻率和汤姆孙系数，:math:`J` 是电流密度，
需要求解 :math:`T(x)` 温度分布。
显然地，当材料性能都为常数时，方程 :eq:`governing_eq`
就是一个基本的二阶常微分方程，非常容易给出解析解。
然而，当材料性质是随着温度变化时，上面的方程就变得非常复杂了。
刘等人就是提出了一种办法可以处理这个复杂问题的办法，
即所谓的工程热电性能的概念。该理论中，
发电器件的能量转化效率和输出功率可以表达为:

.. math::
    :label: eng_yita_gen

    \eta = \eta _c \frac{
        \sqrt{1+ZT_{eng} \alpha_1 \eta _c^{-1}} -1
    }{
        \alpha_0 \sqrt{1+ZT_{eng} \alpha_1 \eta _c^{-1}} + \alpha_2
    }

.. math::
    :label: eng_pd_gen

    P_d = \frac{PF_{eng} \Delta T}{4L}

这里 :math:`\eta_c = (T_h-T_c)/T_h` 是卡诺效率，
:math:`T_h` 和 :math:`T_c` 分别是高、低温端的温度，
:math:`ZT_{eng}` 和 :math:`PF_{eng}` 
即为工程热电优值和工程功率因子，
:math:`\alpha_i(i=0,1,2)`
是一些无量纲的与材料性质相关的系数，
具体地可以参考原始的文献资料。
结论性地，我们会知道当材料性质温度无关时，
:eq:`eng_yita_gen` 和 :eq:`eng_pd_gen`
会完全退化到经典的公式，与经典公式完全相容。

实际上，该理论中还给出了任意电流和截面比条件下的功率输出和热流输入，
比如对于热电单腿:

.. math::
    :label: eng_pout_gen

    P_{out} = \frac{V_{oc}^2}{R} \frac{m}{(1+m)^2}

.. math::
    :label: eng_qhot_gen

    Q_{hot} = \frac{A}{L} \int_{T_c}^{T_h}\kappa(T)dT
        + I T_h S(T_h) - W_J I^2 R
        - W_T I \int_{T_c}^{T_h}\tau (T)dT

其中 :math:`m = R_L / R` 为负载电阻 :math:`R_L` 
材料内阻 :math:`R` 的比值，:math:`V_{oc}` 为开路电压，
:math:`I` 为工作电流，:math:`A` 和 :math:`L`
分别为热电腿的截面积和高度，
:math:`W_J` 和 :math:`W_T` 为焦耳热和汤姆孙热权重系数:

.. math::
    :label: eng_props

    R &= \frac{1}{\Delta T} \frac{L}{A} \int_{T_c}^{T_h}\rho (T)dT \\
    V_{oc} &= \int_{T_c}^{T_h} S (T)dT \\
    W_J &= \frac{
        \int_{T_c}^{T_h} \int_{T}^{T_h} \rho (T)dTdT
    }{
        \Delta T \int_{T_c}^{T_h}\rho (T)dT
    }\\
    W_T &= \frac{
        \int_{T_c}^{T_h} \int_{T}^{T_h} \tau (T)dTdT
    }{
        \Delta T \int_{T_c}^{T_h}\tau (T)dT
    }

在这个基础上我们就可以实现任意电流和尺寸器件的性能预测。

程序实现
--------

这部分的程序实现就是严格按照定义公式进行计算。
通常情况下，可以调用命令 ``tef-engout``
进行工程热电性能的计算，
它可以给出最优电流和截面比下的输出功率和转化效率。
对于更加复杂的功能，比如完整的性能与电流关系曲线，
就需要调用相关的模块。

命令行指令
^^^^^^^^^^

我们可以通过 ``tef-engout -h`` 选项来查看帮助，

.. code-block::

    $ tef-engout -h
    usage: tef-engout [-h] [-b] [-p] [-R] [-L LENGTH] [-s SUFFIX] INPUTFILE [OUTPUTFILE]
    
    Calculate engineering thermoelectric performance - TEflow(0.0.1a3)
    
      >>> Prepare a data file with columns in 'TCSK' order for single-leg
      devices, or 'TCSKCSK' for paired-leg devices if the -p (--pair) option is
      enabled. The conductivity column can accept resistivity values using the
      -R (--resistivity) option. Currently, only evaluations for thermoelectric
      generators are supported.
    
    positional arguments:
      INPUTFILE             Input file name (must be provided)
      OUTPUTFILE            Output file name (optional, auto-generated if omitted)
    
    optional arguments:
      -h, --help            show this help message and exit
      -b, --bare            Output data without header
      -p, --pair            Enable the two-leg model, otherwise the single-leg model.
      -R, --resistivity     Use resistivity instead of the default conductivity.
      -L LENGTH, --length LENGTH
                            The leg length (or height) in mm. Defaults to 1 mm.
      -s SUFFIX, --suffix SUFFIX
                            Suffix for generating the output file name (default: engout)

对于热电单腿，输入文件需要依次包含 4 列数据：
温度(T[K]), 电导率(C[S/cm]), 塞贝克系数(S[uV/K])
和热导率(K[W/(m·K)])。
对于热电单偶，我需要确保两种材料性能的温度取点一致，
输入文件需要依次包含 7 列:
第一列是 T, 然后是第一个材料的 C, S, 和 K, 
再后面是另外一个材料的 C, S, 和 K。
假定数据文件名为 data.txt,
计算单腿器件性能:

.. code-block:: bash

    $ tef-engout data.txt

计算双腿器件性能:

.. code-block:: bash

    $ tef-engout --pair data.txt

相关的模块
^^^^^^^^^^

我们提供了比较完整的函数和类文档，具体可以参考
:doc:`engout </api_doc/teflow.engout>`
模块。


