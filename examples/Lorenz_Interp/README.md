# Demo: Solve Lorenz Number & Data Interpolation

## Step 1 - Solve Lorenz Number

Here we provide an example script, [cal_L.py](./cal_L.py),
which uses the Single Parabolic Band (SPB) model and
the Single Kane Band (SKB) model to determine the Lorenz number
based on Seebeck coefficient. First, we solve
the Fermi level from the known Seebeck coefficient, then 
determine the Lorenz number from this Fermi level.
It's worth noting that in the SPB model, the Lorenz number
can be uniquely determined by the Seebeck coefficient.
In contrast, in the SKB model, the Lorenz number also
depends on the temperature and the bandgap. In this example,
we set the bandgap to 0.5 eV, and the temperature ranges
from 300 to 800 K, with an interval of 100 K.
Run the script as follows:

```bash
python3 cal_L.py
```

If executed successfully, an output file `Lorenz_number.txt`
will be generated, which theoretically should match the contents of
[Lorenz_number-check.txt](./Lorenz_number-check.txt).
Its first column represents the Seebeck coefficient,
the second column represents the Lorenz number under the SPB model,
and from the third column to the end are the Lorenz numbers
at different temperatures under the SKB model.

**Note**: The current program input considers the absolute
value of the Seebeck coefficient. This means that,
regardless of whether it's an n-type or p-type semiconductor,
a positive Seebeck coefficient should be entered.
Otherwise, undefined and uncontrollable errors might occur.

## Step 2 - Data Interpolation

We previously set the Seebeck coefficient values ranging
from 2 to 400 uV/K with an interval of 2 uV/K.
We can obtain the Lorenz number corresponding to the experimental
Seebeck coefficient through data interpolation.
The test data for this part comes from the file
[Seebeck-check.txt](./Seebeck-check.txt).
Execute the following:

```bash
cp Seebeck-check.txt Seebeck.txt
tef-interp -m cubic Lorenz_number.txt Seebeck.txt
```

Here, the `-m` option specifies the interpolation method as `cubic`,
which means cubic spline interpolation (if not specified,
`linear` interpolation is used by default).
The first file specifies the reference source for interpolation,
which is the data we just generated.
The second file contains only one column of data,
which represents the points we want to interpolate.
After running the above commands, the output will overwrite our
`Seebeck.txt` file. Upon inspection, you'll find that it now
contains the results for the Lorenz number,
which theoretically should match the contents of
[Seebeck_interp-check.txt](./Seebeck_interp-check.txt).
