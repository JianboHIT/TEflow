# 演示：工程热电性能计算

为了评估热电单腿或单对热电器件的工程热电性能，需要准备材料的基本性质数据：
温度点（单位：K）、电导率（单位：S/cm）、塞贝克系数（单位：μV/K）
和热导率（单位：W/(m·K)）。同时，也支持使用电阻率（单位：μΩ·m）来替代电导率，
需要配合 `-R` (等价于 `--resistivity`) 选项使用。

**注意：** 为了确保其中积分计算的稳定性，有必要对通常的实验数据进行密集插值。
一般来说，建议插值后大约接近100个点。`tef-interp` 命令可以在这方面提供一些便利。

## 热电单腿

对于热电单腿，我们需要准备一个数据文件，它按顺序包含以下 4 列数据：
温度、电导率（或电阻率）、塞贝克系数和热导率。可以参照我们这里的示例文件
[datas_n.txt](./datas_n.txt)。这个文件中，由于是默认期望的电导率数据，
因此我们可以直接运行如下的命令且不需要任何额外的选项：

```bash
tef-engout datas_n.txt
```

运行成功后，将生成一个名为 `datas_n_engout.txt` 的文件，
里面就是我们计算的工程热电性能结果，包括：
工程功率因子 *PF<sub>eng</sub>*（单位：W/(m·K)）、
工程热电优值 *ZT<sub>eng</sub>*（无量纲）、
输出功率密度 *P<sub>out</sub>*（单位：W/cm²）
和热电转换效率 *Yita*（单位：%）。
该文件的内容应该与文件
[datas_n_engout-check.txt](./datas_n_engout-check.txt)
内容一样。

## 单对热电器件

在计算 $\pi$ - 型热电器件的工程热电性能时，需要将两腿的所有性质插值到相同的温度点，
再将两种材料的性质数据合并到一个文件中，该文件需要依次包含包含 7 列数据：
第一列为温度，然后三列为第一种材料性质（按电导率/电阻率、塞贝克系数和热导率的顺序），
再三列为第二种材料的性质（和前面顺序一样）。可以参照我们这里的示例文件
[datas_np.txt](./datas_np.txt)。在这个文件中，使用了电阻率数据而不是默认的电导率，
也是为了示范一下 `-R` (`--resistivity`) 选项的使用。然后，通过添加 `--pair`
选项来指定是进行单对热电器件的计算：

```bash
tef-engout -R --pair datas_np.txt
```

计算完成后，将生成一个名为 `datas_np_engout.txt` 的文件，
里面就是我们计算的工程热电性能结果。该文件的内容应该与文件
[datas_np_engout-check.txt](./datas_np_engout-check.txt)
内容一样。