#!/usr/bin/env python3

import os
import numpy as np
from pprint import pformat
from TEoutput import ztdev
from TEoutput.utils import get_root_logger


############# Default Setting #############     
filedatas = 'ztdev_datas.txt'
fileoutput = 'ztdev_output.txt'
comment = ''
###########################################

# config logger
LEVELNUM = 20
logger = get_root_logger()
logger.setLevel(LEVELNUM)

sname = '{} - {}'.format(logger.name, os.path.basename(__file__))
logger.info('Calcuate device ZT (ZTdev): %s', sname)

# read input datas
datas = np.loadtxt(filedatas, unpack=True, ndmin=2)
logger.info('Read property datas from file %s', filedatas)
data_spot = pformat(datas[:,::5].T)
logger.debug('Data spot checking:\n%s', data_spot)

# perform calculation
logger.info('Perform calculation using ztdev.valuate()')
out = ztdev.valuate(datas)
outdata = np.vstack([out.deltaT, out.ZTdev, out.Yita]).T
logger.info('Export results: deltaT, ZTdev, Yita')
data_spot = pformat(outdata[::5,:])
logger.debug('Data spot checking:\n%s', data_spot)

# output result
if comment == '':
    from datetime import datetime
    date = datetime.now().strftime("%Y.%m.%d")
    comment = 'Calculate device ZT ({})'.format(date)
    logger.debug('Automatically generate comment:\n%s', comment)
else:
    logger.debug('Read comment info.\n%s', comment)
comment += '\ndeltaT[K] ZTdev[1] Yita[%]'
np.savetxt(fileoutput, outdata, fmt='%10.4f', header=comment)
logger.info('Save results to %s. (Done)', fileoutput)
