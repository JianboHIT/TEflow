#!/usr/bin/env python3

import os
import numpy as np
from teflow.bandlib import APSSPB, APSSKB
from teflow.utils import get_root_logger


# config logger for the screen display information
LEVELNUM = 20
logger = get_root_logger(level=LEVELNUM)
sname = f'{logger.name} - {os.path.basename(__file__)}'
logger.info(f'Calcuate Lorenz number: {sname}')

# parameters
Eg = 0.5                              # Bandgap in eV
T = [300, 400, 500, 600, 700, 800]    # Temperature in K
S_ref = np.linspace(2, 400, 200)      # Seebeck coefficient in uV/K
fileoutput = 'Lorenz_number.txt'
logger.info(f'Bandgap is set at {Eg} eV')

# solve L from S using SPB model
#    In the SPB model, the Lorenz number depends solely on the Seebeck
#    coefficient. Therefore, the temperature is set to 300 K arbitrarily.
spb = APSSPB()
Fermi = spb.solve_EF('S', S_ref, 300)
L_spb = spb.L(Fermi, 300)

out = [S_ref, L_spb]
info = f'S(Eg_{Eg}eV)   L-SPB'
logger.info('Solve L from S using SPB model')

# solve L at each T using SKB model
#    In the SKB model, the Lorenz number is influenced not just by the
#    Seebeck coefficient but also by temperature and the bandgap. We
#    typically start by using the Seebeck coefficient to gauge the Fermi
#    level. Then, based on that, we determine the Lorenz constant.
for Temp in T:
    skb = APSSKB(Eg=Eg)
    Fermi = skb.solve_EF('S', S_ref, Temp)
    dataL = skb.L(Fermi, Temp)
    out.append(dataL)
    info += f' L-SKB_{Temp}K'
    logger.info(f'Solve L from S at T={Temp} K using SKB model')

# write results to file
out = np.transpose(np.array(out))
np.savetxt(fileoutput, out, fmt='%10.4f', header=info)
logger.info(f'Save results to {fileoutput} (DONE)')
