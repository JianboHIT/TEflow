========
快速上手
========

程序安装
--------

此软件包核心只依赖 python3.8+ 环境和 numpy/scipy 库，
我们可以通过 pip 直接安装所需要的库：

.. code-block:: bash

    $ pip install teflow

确保网络连接正常，依赖的库会自动安装。
再或者，我们也可以选择从源码安装，预览新功能。
源码发布在 Github / Gitee 上。

.. attention::

    python3.7 已经在 2023 年 6 月停止了维护更新，
    我们的程序也都是在 python3.8 的环境下开发和测试，
    虽然部分功能在更低的版本上可能可以运行，
    但我们并不打算保证低版本的兼容性，或修复由此导致的问题。
    而且，在测试中我们发现相比 python3.6, 我们的程序包
    在 python3.8 中可以提速 2 ~ 5 倍，
    因此我们强烈建议用户放弃在低版本 python 上使用该程序的计划。

命令行操作
----------

在正确安装之后，我们可以通过命令行来使用 TEflow 程序（缩写为 ``tef`` ）。
在 Linux 系统下进行命令操作一个非常自然的事情，
在 Windows 或其它平台下，需要首先进入终端，然后进行命令操作。
一般地，我们可以通过 -h (或 --help) 选项来显示帮助信息。
如果 TEflow 成功安装的话，
当我们运行 ``tef -h`` 之后，应该会获得类似下面的显示信息
(后续随着软件发展可能显示内容可能会有不同):

.. code-block::

    $ tef -h

          ________  ________
         |        \|        \     ,...  ,,
          \$$$$$$$$| $$$$$$$$   .d' ""`7MM
            | $$   | $$__       dM`     MM
            | $$   | $$  \     mMMmm    MM  ,pW"Wq.`7M'    ,A    `MF'
            | $$   | $$$$$      MM      MM 6W'   `Wb VA   ,VAA   ,V
            | $$   | $$_____    MM      MM 8M     M8  VA ,V  VA ,V
            | $$   | $$     \   MM      MM YA.   ,A9   VVV    VVV
             \$$    \$$$$$$$$ .JMML.  .JMML.`Ybmd9'     W      W
    
                             (v0.0.1a3, Copyright 2023 Jianbo ZHU)
    ______________________________________________________________________
    >>> Streamline your thermoelectric workflow from materials to devices
    
    Usage: tef-subcommand [-h] ...
    
    Subcommands:
        format  Format thermoelectric properties data
          band  Insight carriar transport with band models
         ztdev  Calculate ZTdev of thermoelectric generator
        interp  Data interpolation and extrapolation
        mixing  Mixing the datafile with same array-shape
        refine  Remove redundant & extract the concerned data
        cutoff  Cut-off data at the threshold temperature

我们可以通过 ``tef-xxxxxx -h`` 来详细查看子命令 ``xxxxxx`` 的用法和支持选项。
比如:

.. code-block::

    $ tef-ztdev -h
    usage: tef-ztdev [-h] [-b] [-y] [-s SUFFIX] INPUTFILE [OUTPUTFILE]
    
    Calculate ZTdev of thermoelectric generator - TEflow(0.0.1a3)
    
      >>> Prepare a data file with columns in the following order: T, C, S, and
      K. Alternatively, you can format the data file with Tc, Th, and a pre-
      calculated Yita; in this case, include the -y (or --yita) option to
      indicate the use of this format.
    
    positional arguments:
      INPUTFILE             Input file name (must be provided)
      OUTPUTFILE            Output file name (optional, auto-generated if omitted)
    
    optional arguments:
      -h, --help            show this help message and exit
      -b, --bare            Output data without header
      -y, --yita            Use this option when the input file is formatted with
                            columns Tc, Th, Yita
      -s SUFFIX, --suffix SUFFIX
                            Suffix for generating the output file name (default: ztdev)

为了计算热电材料的工程 ZT 值，我们需要准备一个输入文件，
里面包含 4 列材料属性数据，依次是温度 (K)，电导率 (S/cm)，
塞贝克系数 (uV/K)，和热导率 (W/(m.K))。
在 Github / Gitee 仓库的 examples/ZTdev 文件中有更详细的描述。
然后，我们运行如下命令就可以进行计算
(假设数据文件名为 data.txt):

.. code-block:: bash

    $ tef-ztdev data.txt

参照屏幕显示，输出结果会被保存在文件 ``data_ztdev.txt`` 。
当然，按照前面显示的帮助信息，我们可以自己指定输出文件名。
比如我们希望结果保存在 ``ZTdev-S1.txt``,
我们可以运行如下命令:

.. code-block:: bash

    $ tef-ztdev data.txt ZTdev-S1.txt

我们可以进一步利用其它的选项做更加复杂的事情。

用作第三方库
------------

如果你需要集成 TEflow 程序包的功能到你自己的程序，
比如批量化或者流程化数据处理时，
最直接的方式就是调用 **valuate()** 方法。
对于绝大多数子模块或者类，我们一般会提供一个名为 **valuate()**
的函数或者方法，它会在通用的默认参数下调用相关的计算。
比如，我们想计算一系列材料的工程 ZT 值，准备了相应的数据文件，
文件名为 data-S1.txt, data-S2.txt, ..., data-S10.txt ，
我们可以通过下面的方式来直接计算所有数据:

.. code-block:: python

    >>> import numpy as np
    >>> from teflow import ztdev
    >>> 
    >>> for i in range(10):
    ...     data = np.loadtxt(f'data-S{i+1}.txt', unpack=True)
    ...     rst = ztdev.valuate(data)
    ...     output = np.c_[rst['Tc'], rst['Th'],rst['Yita'],rst['ZTdev']]
    ...     np.savetxt(f'output-S{i+1}.txt', output)
    ... 
    >>>

在这个代码片段中，我们采用了 numpy.loadtxt() 方法来读取数据
(这里 ``unpack=True`` 是非常重要的，它确保数据的每一列被单独的存储；
而 ``i+1`` 被使用是由于 python 的索引是 0 开始的),
然后调用 ztdev.valuate() 方法进行计算，
计算结果以字典形式保存，
最后我们整理格式并保存到相应的文件。

在程序包中，我们还提供了其它更多的功能，比如热电势的计算等。
我们为每一个函数和类都写了详细的文档，可以直接查看源码，
也可以查看本文档的 :doc:`api_doc/index` 章节。
