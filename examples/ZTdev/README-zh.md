# 演示：器件热电优值 (ZT<sub>dev</sub>)

当我们计划从材料性能计算 ZT<sub>dev</sub> 时，
我们需要准备一个类似 [Data_PbTe.txt](./Data_PbTe.txt) 的文件，
其中包含四列，依次是温度（单位：K）、电导率（单位：S/cm）、塞贝克系数（单位：uV/K）
和热导率（单位：W/(m.K) ）。然后运行如下命令：

```bash
tef-ztdev Data_PbTe.txt
```

成功执行后，将生成名为 `Data_PbTe_ztdev.txt` 的文件，里面包含了转化效率和 ZT<sub>dev</sub> ，
内容应该和 [ZTdev_PbTe-check.txt](./ZTdev_PbTe-check.txt) 一样。

除此以外，我们还可以通过其它方式预先获得的转化效率来估算一个材料或者器件的 ZT<sub>dev</sub> 。
此时我们需要准备一个类似 [Yita_PbTe.txt](./Yita_PbTe.txt) 的文件，其中包含三列，依次是：
低温端温度（单位：K）、高温端温度（单位：K）和相应的转化效率（以 % 为单位，范围从0到100）。
然后运行如下命令：

```bash
tef-ztdev -y Yita_PbTe.txt
```

成功执行后，将生成名为 `Yita_PbTe_ztdev.txt` 的文件，里面会包含 ZT<sub>dev</sub>，
其内容也应该和 [ZTdev_PbTe-check.txt](./ZTdev_PbTe-check.txt) 一样。
