#!/usr/bin/env python3

import os
import numpy as np
from TEoutput.engout import GenPair
from TEoutput.utils import get_root_logger


############# Default Setting #############     
fileinput_p = 'gen_input_p.txt'
fileinput_n = 'gen_input_n.txt'
fileoutput = 'gen_output.txt'
height = 1    # unit: mm
comment = ''
###########################################

# config logger
LEVELNUM = 20
logger = get_root_logger(level=LEVELNUM)

# header
sname = '{} - {}'.format(logger.name, os.path.basename(__file__))
logger.info('Calcuate engineering performance of generator: %s', sname)

# read input datas_p
datas_p = np.loadtxt(fileinput_p, unpack=True, ndmin=2)
logger.info('Read property datas of p-type leg from file %s', fileinput_p)
data_spot = np.array_str(datas_p[:,::10].T, max_line_width=85)
logger.debug('Data spot checking with step=10:\n%s', data_spot)

# read input datas_n
datas_n = np.loadtxt(fileinput_n, unpack=True, ndmin=2)
logger.info('Read property datas of n-type leg from file %s', fileinput_n)
data_spot = np.array_str(datas_n[:,::10].T, max_line_width=85)
logger.debug('Data spot checking with step=10:\n%s', data_spot)

# perform calculation
logger.info('Perform simulating of thermoelectric generator.')
out = GenPair.valuate(datas_p, datas_n, L=height)
props = ['deltaT', 'PFeng', 'ZTeng', 'Pout', 'Yita']
outdata = np.vstack([out[prop] for prop in props]).T
props[3] = 'Pd'
logger.info('Export results: %s', ', '.join(props))
data_spot = np.array_str(outdata[::10,:], max_line_width=85)
logger.debug('Data spot checking with step=10:\n%s', data_spot)

# output result
if comment == '':
    from datetime import datetime
    date = datetime.now().strftime("%Y.%m.%d")
    dsp = 'engineering output of generator with height {} mm ({})'
    comment = dsp.format(height, date)
    logger.debug('Automatically generate comment:\n%s', comment)
else:
    logger.debug('Read comment info.\n%s', comment)
comment += '\ndeltaT[K] PFeng[W/m.K] ZTeng[1] Pd[W/cm^2] Yita[%]'
np.savetxt(fileoutput, outdata, fmt='%10.4f', header=comment)
logger.info('Save results to %s. (Done)', fileoutput)

