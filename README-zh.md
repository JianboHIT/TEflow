# TEflow 程序包（开发中）
一个用于简化从材料到器件工作流程的 Python3 程序包

## 功能
- 载流子传输模型
  - 单抛物带（SPB）模型
  - 单Kane带（SKB）模型
  - 多带模型
  - 自定义带模型
- 工程热电性能[^1]
  - 工程无量纲优值（ZT<sub>eng</sub>）和功率因子（PF<sub>eng</sub>）
  - 最大效率（η<sub>max</sub>）和输出功率密度（P<sub>d</sub>）
  - 器件性能（例如输出电压 V 、热流量 Q<sub>hot</sub>）随负载电阻
    R<sub>L</sub> （或电流密度 I）的完整变化曲线
- 器件热电优值 ZT<sub>dev</sub>[^2]
  - 最大热电器件效率
  - 优化的相对电流密度 $u$
  - 热电势 $\Phi$ 计算
- 热电数据处理
  - 热电数据的插值和外推
  - 在阈值温度处截断热电数据
  - 拼接和重新排列平行的数据文件
  - 线性加权组合平行的数据文件

<br/><br/>
#### 参考文献

[^1]: Kim, H. S., Liu, W., Chen, G., Chu, C. W., & Ren, Z. (2015). Relationship between thermoelectric figure of merit and energy conversion efficiency. 
_Proceedings of the National Academy of Sciences_, 112(27), 8205-8210. DOI: [10.1073/pnas.1510231112](https://doi.org/10.1073/pnas.1510231112)

[^2]: Snyder, G. J., & Snyder, A. H. (2017). Figure of merit ZT of a thermoelectric device defined from materials properties. 
_Energy & Environmental Science_, 10(11), 2280-2283. DOI: [10.1039/C7EE02007D](https://doi.org/10.1039/C7EE02007D)
