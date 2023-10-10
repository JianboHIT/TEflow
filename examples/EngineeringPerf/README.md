# Demo: Engineering Thermoelectric Performance Calculation

To evaluate the engineering thermoelectric performance of either a
single-leg or a unicouple device, one must prepare temperature-dependent
material property data. This should include the sampling temperature
(in Kelvin), electrical conductivity (in S/cm), Seebeck coefficient
(in μV/K), and thermal conductivity (in W/(m·K)). Alternatively,
you can use resistivity (in μΩ·m) in place of electrical conductivity
in the input file, along with the `-R` (or `--resistivity`) flag.

**Note:** To ensure the stability of the integration calculations,
it is essential to densely interpolate the experimental data.
Typically, a benchmark of around 100 temperature points is suggested.
The `tef-interp` command might offer some convenience in this regard.

## Thermoelectric Leg

For single-leg thermoelectric devices, we need to prepare a data file
with 4 columns in the following order: temperature, electrical conductivity
(or resistivity), Seebeck coefficient, and thermal conductivity.
Refer to the example file [datas_n.txt](./datas_n.txt) for guidance.
In this data file, electrical conductivity is prepared,
allowing you to run the command without any additional options,
as follows:

```bash
tef-engout datas_n.txt
```

After successful execution, a file named `datas_n_engout.txt` will be
generated. This file will contain the desired engineering performance
of the thermoelectric leg:
engineering power factor *PF<sub>eng</sub>* (in W/(m·K)),
engineering figure-of-merit *ZT<sub>eng</sub>* (dimensionless),
output power *P<sub>out</sub>* (in W/cm²),
and conversion efficiency *Yita* (in %).
The content of this file should align with what is found in
[datas_n_engout-check.txt](./datas_n_engout-check.txt).

## Thermoelectric Unicouple

When evaluating the engineering thermoelectric performance of a $\pi$-shaped
unicouple, it is required that all properties of both legs share
the same sampling temperature. Then, combine the property data of the
two materials into a single file with 7 columns: the first for temperature,
following three for the first material (in the order of electrical
conductivity/resistivity, Seebeck coefficient, and thermal conductivity),
and the next three for the second material in the same order.
Refer to the example file [datas_np.txt](./datas_np.txt) for guidance.
In this file, resistivity data is used instead of conductivity,
showcasing the `-R` (or `--resistivity`) option.
Enable the `--pair` option to indicate a calculation task for
a thermoelectric unicouple:

```bash
tef-engout -R --pair datas_np.txt
```

Upon successful execution, a file named `datas_np_engout.txt` will be
generated, containing the expected engineering performance.
The content of this file should align with what is found in
[datas_np_engout-check.txt](./datas_np_engout-check.txt).
