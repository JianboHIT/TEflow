#!/usr/bin/env python3

import logging
import numpy as np
from teflow.utils import interp


############# Default Setting #############
fileinput = 'interp_input.txt'      
fileoutput = 'interp_output.txt'
method = 'linear'
npoints = 101
comment = ''
###########################################


# TODO: help doc
# TODO: argparse

# valid method to interpolate
KINDS = {'linear', 'nearest', 'nearest-up', 
         'zero', 'slinear', 'quadratic', 'cubic', 
         'previous', 'next',
         'poly1', 'poly2','poly3','poly4','poly5',}

# config logging
LEVEL = logging.DEBUG
FORMAT = '[%(levelname)s] %(message)s'
logging.basicConfig( format=FORMAT, level=LEVEL)

# read origin data
x, *y = np.loadtxt(fileinput, unpack=True, ndmin=2)
logging.info('Read data_x and data_y from {}.'.format(fileinput))

# read sampling points
try:
    x2, *_ = np.loadtxt(fileoutput, unpack=True, ndmin=2)
except IOError:
    # failed to read sampling points and set them automatically
    logging.info('Failed to read sampling points from {}.'.format(fileoutput))
    
    x2 = np.linspace(x[0], x[-1], num=npoints)
    
    logging.info('Generate sampling points automatically.')
    logging.debug('Using np.linspace({}, {}, num={}) .'.format(x[0], x[-1], npoints))
except Exception as err:
    # catch other error
    logging.error('Failed to read/generate sampling points.')
    raise(err)
else:
    logging.info('Read sampling points from {}.'.format(fileoutput))
finally:
    logging.debug('Get sampling successfully.')

# check method and do interpolate
method = method.lower()
if method not in KINDS:
    logging.error('Failed to recognize the method of interpolation.')
    dsp = 'Now is {}, but methods shown below are allowed: \n{}'
    logging.error(dsp.format(method, KINDS))
    raise ValueError("Value of 'method' is invalid.")
else:
    datas = interp(x, y, x2, method=method, merge=True)
    logging.info('Perform the interpolation operation.')

# data result
if comment == '':
    from datetime import datetime
    date = datetime.now().strftime("%Y.%m.%d")
    comment = 'interpolate via {} method ({})'.format(method, date)
    logging.debug('Automatically generate comment:\n{}'.format(comment))
else:
    logging.debug('Read comment info.\n{}'.format(comment))

# save result
np.savetxt(fileoutput, datas.T, fmt='%.4f', header=comment)
logging.info('Save interpolation results to {}. (Done)'.format(fileoutput))
