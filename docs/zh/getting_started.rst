========
快速上手
========

程序安装
--------

此软件包只依赖 python3 环境和 numpy/scipy 库，可以通过 pip 直接安装：

.. code-block:: bash

    $ pip install teflow

确保网络连接正常，应该会正常安装，所有其包含的功能都会可用。
我们后续可能会为一些其它与热电相关的库提供便捷使用接口，
会有更多的依赖选项，但会始终保证如此安装后基本功能都可用。

命令行操作
----------

在正确安装之后，我们可以通过命令行来调用程序。我们可以通过下面的命令查看可用
的命令选项(后续随着软件发展可能显示内容会有不同):

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

    Usage: tef-xxxxxx ...

    Subcommands:
        interp    Data interpolation and extrapolation
        mixing    Mixing the datafile with same array-shape
        ztdev    Calculate ZTdev of thermoelectric generator
        format    Format thermoelectric properties data
        cutoff    Cut-off data at the threshold temperature
        refine    Remove redundant & extract the concerned data

然后我们可以通过 `tef-xxxxxx -h` 来详细查看命令的选项。
