#!/usr/bin/env python3

import numpy as np
from TEoutput.utils import interp


############# Default Setting #############
fileorigin = 'interp_origin.txt'      
filedatas = 'interp_datas.txt'
method = 'linear'
npoints = 101
comment = ''
###########################################


# TODO: help doc
# TODO: logging
# TODO: argparse

# valid method to interpolate
KINDS = {'linear', 'nearest', 'nearest-up', 
         'zero', 'slinear', 'quadratic', 'cubic', 
         'previous', 'next',
         'poly1', 'poly2','poly3','poly4','poly5',}

# read origin data
x, *y = np.loadtxt(fileorigin, unpack=True)

# read new points
try:
    x2, *_ = np.loadtxt(filedatas, unpack=True)
except IOError:
    # failed to read sampling points and set them automatically
    x2 = np.linspace(x[0], x[-1], num=npoints)
except Exception as err:
    # catch other error
    raise(err)
else:
    # logging
    pass

# check method and do interpolate
method = method.lower()
if method not in KINDS:
    raise ValueError("Value of 'method' is invalid.")
else:
    datas = interp(x, y, x2, method=method, merge=True)

# data result
if comment == '':
    from datetime import datetime
    date = datetime.now().strftime("%Y.%m.%d")
    comment = 'interpolate via {} method ({})'.format(method, date)
np.savetxt(filedatas, datas.T, fmt='%.4f', header=comment)