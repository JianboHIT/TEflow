#!/usr/bin/env python3

import logging
import numpy as np
from pprint import pformat
from TEoutput.engout import Generator
from TEoutput.utils import get_root_logger


############# Default Setting #############     
filedatas = 'gen_datas.txt'
fileoutput = 'gen_output.txt'
height = 1    # unit: mm
comment = ''
###########################################

# config logging
LEVEL = logging.DEBUG
logger = get_root_logger()
logger.setLevel(LEVEL)

sname = '{} - eng_valuate_gen.py'.format(logger.name)
logger.info('Calcuate engineering performance of generator: {}'.format(sname))

# read input datas
datas = np.loadtxt(filedatas, unpack=True, ndmin=2)
logger.info('Read property datas from file {}'.format(filedatas))
data_spot = pformat(datas[:,::10].T)
logger.debug('Data spot checking:\n{}'.format(data_spot))

# perform calculation
logger.info('Perform simulating of thermoelectric generator.')
out = Generator.valuate(datas, L=height)  # PFeng, ZTeng, Pd, Yita
outdata = np.vstack(out).T
logger.info('Export results: deltaT, PFeng, ZTeng, Pd, Yita')
data_spot = pformat(outdata[::10,:])
logger.debug('Data spot checking:\n{}'.format(data_spot))

# output result
if comment == '':
    from datetime import datetime
    date = datetime.now().strftime("%Y.%m.%d")
    dsp = 'engineering output of generator with height {} mm ({})'
    comment = dsp.format(height, date)
    logger.debug('Automatically generate comment:\n{}'.format(comment))
else:
    logger.debug('Read comment info.\n{}'.format(comment))
comment += '\ndeltaT PFeng ZTeng Pd Yita'
np.savetxt(fileoutput, outdata, fmt='%.4f', header=comment)
logger.info('Save interpolation results to {}. (Done)'.format(fileoutput))
