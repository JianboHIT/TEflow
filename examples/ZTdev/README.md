# Demo: Device Figure-of-Merit (ZT<sub>dev</sub>)

To calculate ZT<sub>dev</sub> based on material properties,
you should prepare a file similar to [Data_PbTe.txt](./Data_PbTe.txt).
This file must contain four columns in the following order: temperature (in K),
electrical conductivity (in S/cm), Seebeck coefficient (in uV/K),
and thermal conductivity (in W/(m·K)). Then, execute the following command:

```bash
tef-ztdev Data_PbTe.txt
```

Upon successful execution, a file named `Data_PbTe_ztdev.txt` will be generated.
This file will detail the conversion efficiency and ZT<sub>dev</sub>.
The content of this file should align with what is found in
[ZTdev_PbTe-check.txt](./ZTdev_PbTe-check.txt).

For those looking to estimate the ZT<sub>dev</sub> of a material or a device
using pre-obtained conversion efficiencies, a file similar to
[Yita_PbTe.txt](./Yita_PbTe.txt) is required. This file should have three columns:
the cold end temperature (in K), the hot end temperature (in K),
and the corresponding conversion efficiency (expressed in % and ranging from 0 to 100).
Then, run the following command:

```bash
tef-ztdev -y Yita_PbTe.txt
```

If the command runs successfully, a file named `Yita_PbTe_ztdev.txt` will be produced,
which contains the ZT<sub>dev</sub> values. The content of this file should also
align with what is found in [ZTdev_PbTe-check.txt](./ZTdev_PbTe-check.txt).
