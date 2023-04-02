#!/usr/bin/env python3

import os
import numpy as np
from teflow.engout import GenLeg
from teflow.utils import get_root_logger


############# Default Setting #############     
fileinput = 'gen_input.txt'
fileoutput = 'gen_output.txt'
height = 1    # unit: mm
comment = ''
###########################################

# config logger
LEVELNUM = 20
logger = get_root_logger()
logger.setLevel(LEVELNUM)

sname = '{} - {}'.format(logger.name, os.path.basename(__file__))
logger.info('Calcuate engineering performance of generator: %s', sname)

# read input datas
datas = np.loadtxt(fileinput, unpack=True, ndmin=2)
logger.info('Read property datas from file %s', fileinput)
data_spot = np.array_str(datas[:,::10].T, max_line_width=85)
logger.debug('Data spot checking with step=10:\n%s', data_spot)

# perform calculation
logger.info('Perform simulating of thermoelectric generator.')
out = GenLeg.valuate(datas, L=height)
props = ['deltaT', 'PFeng', 'ZTeng', 'Pout', 'Yita']
outdata = np.vstack([out[prop] for prop in props]).T
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

