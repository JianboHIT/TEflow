#   Copyright 2023 Jianbo ZHU
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
from datetime import datetime

from ._version import __version__
from .utils import get_pkg_name, get_root_logger

CMD = 'teop'
PKG = get_pkg_name()
VISION = __version__
INFO = f'{PKG}({VISION})'
TIME = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
DESCRIPTION = {
    'interp': 'Data interpolation and extrapolation',
    'mixing': 'Mixing the datafile with same array-shape',
    'ztdev' : 'Calculate ZTdev of thermoelectric generator',
    'cutoff': 'Cut-off data at the threshold temperature',
    'refine': 'Remove all comments and blank lines in file'
}

logger = get_root_logger(level=20, fmt='[%(name)s] %(message)s')
# logger = get_root_logger(level=10, fmt='[%(levelname)5s] %(message)s')


def do_main(args=None):
    # for test
    import sys
    if args is None:
        args = sys.argv[1:]
    
    dsp = f"{INFO} @ Python {sys.version}".replace('\n', '')
    if len(args) > 0:
        task = args[0].lower()
        if task.startswith('interp'):
            do_interp(args[1:])
        elif task.startswith('mixing'):
            do_mixing(args[1:])
        elif task.startswith('ztdev'):
            do_ztdev(args[1:])
        elif task.startswith('cutoff'):
            do_cutoff(args[1:])
        elif task.startswith('refine'):
            do_refine(args[1:])
        else:
            print(dsp)
    else:
        print(dsp)


def do_interp(args=None):
    import numpy as np
    from .analysis import interp
    
    task = 'interp'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}', 
        description=f'{DESC} - {INFO}',
        epilog='')
    
    parser.add_argument('inputfile', metavar='INPUTFILE',
                        help='Filename of input file (necessary)')
    
    parser.add_argument('outputfile', metavar='OUTPUTFILE', nargs='?',
                        help='Filename of output file (default: Basename_suffix.Extname)')
    
    parser.add_argument('-b', '--bare', action='store_true',
                        help='Output data without header and X-column')

    parser.add_argument('-m', '--method', default='linear', 
                        help='The method of interpolation and extrapolation \
                             (e.g. linear, cubic, poly3, default: linear)')
    
    parser.add_argument('-e', '--extrapolate', default='auto',
                        help='The strategy to extrapolate points outside the data range \
                             (auto, const, or <value>, default: auto)')
    
    parser.add_argument('-n', '--npoint', type=int, default=101, 
                        help='The number of interpolated data points (default: 101)')

    parser.add_argument('-s', '--suffix', default='interp', 
                        help='The suffix to generate filename of output file (default: interp)')

    # parser.add_argument('-p', '--paired', action='store_true', 
    #                     help='Identify input data in paired format')

    # parser.add_argument('-g', '--group', default='TCSK', 
    #                     help='Group identifiers for paired data (e.g. TCSK, TCTSTK, TKXXTSC, \
    #                          default: TCSK)')
    
    # parser.add_argument('-c', '--cal', action='store_true', 
    #                     help='Calculate thermoelectric power factor and figure-of-merit')

    options = parser.parse_args(args)
    
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    x, *y = np.loadtxt(inputfile, unpack=True, ndmin=2)
    logger.info('Read data_x and data_y from {}.'.format(inputfile))
    
    # parse outputfile name
    if options.outputfile is None:
        name, ext = inputfile.rsplit('.', 1)
        outputfile = f'{name}_{options.suffix}.{ext}'
    else:
        outputfile = options.outputfile
    
    # read sampling points
    try:
        x2, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except IOError:
        # failed to read sampling points and set them automatically
        logger.info(f'Failed to read sampling points from {outputfile}.')
        
        npoint = options.npoint
        x2 = np.linspace(x[0], x[-1], num=npoint)
        
        logger.info('Generate sampling points automatically.')
        logger.debug(f'Using np.linspace({x[0]}, {x[-1]}, num={npoint}) .')
    except Exception as err:
        # catch other error
        logger.error('Failed to read/generate sampling points.\n')
        raise(err)
    else:
        logger.info(f'Read sampling points from {outputfile}.')

    # check method
    _METHODS = {'linear', 'nearest', 'nearest-up', 
             'zero', 'slinear', 'quadratic', 'cubic', 
             'previous', 'next',
             'poly1', 'poly2','poly3','poly4','poly5',}
    method = options.method.lower()
    if method not in _METHODS:
        logger.error('Failed to recognize the method of interpolation.')
        logger.error(f'Now is {method}, but methods shown below are allowed: \n{_METHODS}\n')
        raise ValueError("Value of 'method' is invalid.")
    
    # The strategy to extrapolate points outside the data range
    extrapolate = options.extrapolate.lower()
    toMerge = (not options.bare)
    dsp = 'Perform the interpolation operation with %s extrapolation.'
    if extrapolate.startswith('auto'):
        datas = interp(x, y, x2, method=method, merge=toMerge)
        logger.info(dsp, 'automatic')
    elif extrapolate.startswith('const'):
        y = np.array(y)
        datas = interp(x, y, x2, method=method, merge=toMerge,
                       bounds_error=False,
                       fill_value=(y[:,0], y[:,-1]))
        logger.info(dsp, 'nearest constant')
    else:
        try:
            value = float(extrapolate)
        except Exception as err:
            raise err
        
        datas = interp(x, y, x2, method=method, merge=toMerge, 
                       bounds_error=False,
                       fill_value=value)
        logger.info(dsp, f'specified constant ({extrapolate})')

    # data result
    comment = '' if options.bare else f'Interpolate via {method} method - {TIME} {INFO}'
    np.savetxt(outputfile, datas.T, fmt='%.4f', header=comment)
    logger.info(f'Save interpolation results to {outputfile}. (Done)')
    

def do_mixing(args=None):
    import numpy as np
    from .analysis import mixing
    
    task = 'mixing'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog='')
    
    parser.add_argument('-b', '--bare', action='store_true',
                        help='Output data without header')
    
    parser.add_argument('-w', '--weight', metavar='WEIGHTS',
                        help="Weights of mixing, with the same number of datafiles \
                             (default:'1 1 1 ...')")
    
    parser.add_argument('-s', '--scale', type=float, default=0, 
                        help='The scale factor (default: 1/sum(weights))')
    
    parser.add_argument('-o', '--output', metavar='FILENAME', 
                        help='Specify the filename of output destination. By default, the last \
                             DATAFILE will be treated as output destination')
    
    parser.add_argument('datafile', metavar='DATAFILE', nargs='+',
                        help='Filenames of datafile. Crucially, the last one will be \
                             treated as output destination unless -o/--output option is given.')
        
    options = parser.parse_args(args)
    
    logger.info(f'{DESC} - {TIME}')
    
    if options.output is None:
        *filenames, outputfile = options.datafile
    else:
        filenames = options.datafile
        outputfile = options.output
    logger.info(f"Datafiles: {', '.join(filenames)}")
    logger.info(f'Outputfile: {outputfile}')
    
    datas = [np.loadtxt(filename) for filename in filenames]
    
    if options.weight is None:
        weight = None
        logger.info('Equal weight factors are adopted to mixing datafile')
    elif options.weight.isdecimal():
        weight = list(map(int, options.weight))
        logger.info(f"Weight factors: {' '.join(s for s in options.weight)}")
    else:
        weight = list(map(float, options.weight.split()))
        logger.info(f'Weight factors: {options.weight}')
        
    if abs(options.scale) < 1E-4:
        scale = None
        logger.info(f'The normalized factor is adopted to scale data')
    else:
        scale = options.scale
        logger.info(f'Scale factor: {options.scale}')
    
    data = mixing(datas, weight, scale)
    
    # data result
    comment = '' if options.bare else f'Mixed datafiles - {TIME} {INFO}'
    np.savetxt(outputfile, data, fmt='%.4f', header=comment)
    logger.info(f'Save mixed data to {outputfile}. (Done)')


def do_ztdev(args=None):
    from .ztdev import cal_ZTdev
    
    task = 'ztdev'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog='')
    
    # parser.add_argument('yita', metavar='Yita', type=float,
    #                     help='Efficiency of thermoelectric generator(0 - 100)')
    
    parser.add_argument('-y', '--yita', metavar='EFFICIENCY', type=float, 
                        help='Efficiency of thermoelectric generator (0-100)')
    
    parser.add_argument('tmin', type=float, 
                        help='Temperature at cold side in K')
    
    parser.add_argument('tmax', type=float,
                        help='Temperature at hot side in K')
    
    options = parser.parse_args(args)
    
    logger.info(f'{DESC} - {TIME}')
    
    yita = options.yita
    tmin = options.tmin
    tmax = options.tmax
    logger.info(f'Yita: {yita:.2f} %,  Tmin: {tmin:.2f} K, Tmax: {tmax:.2f} K')
    
    ZTdev = cal_ZTdev(yita, Tc=tmin, Th=tmax)
    logger.info(f'ZTdev: {ZTdev:.4f}. (DONE)')


def do_cutoff(args=None):
    import numpy as np
    
    # # import when necessary
    # from .analysis import boltzmann
    # from .analysis import smoothstep
    
    task = 'cutoff'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog='')
    
    parser.add_argument('t_cut', metavar='T-CUT', type=float,
                        help='Threshold temperature')
    
    parser.add_argument('inputfile', metavar='INPUTFILE',
                        help='Filename of input file (necessary)')
    
    parser.add_argument('outputfile', metavar='OUTPUTFILE', nargs='?',
                        help='Filename of output file (default: Basename_suffix.Extname)')
    
    parser.add_argument('-b', '--bare', action='store_true',
                        help='Output concerned columns, without header')
    
    parser.add_argument('-m', '--method', default='bz',
                        help='The method of cut-off \
                             (Boltzmann[bz], smoothstep[ss], default: bz)')
    
    parser.add_argument('-c', '--column', metavar='COLUMN',
                        help="Specify indexes of column which are tailored (default: '1 2 .. N')")
    
    parser.add_argument('-w', '--width', type=float, default=20,
                        help='The transition width of cut-off function (default: 20)')
    
    parser.add_argument('-s', '--suffix', default='cutoff', 
                        help='The suffix to generate filename of output file (default: cutoff)')
    
    options = parser.parse_args(args)
    # print(options)
    
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    datas = np.loadtxt(inputfile, ndmin=2)
    logger.info('Read datas from {}.'.format(inputfile))
    
    # parse outputfile name
    if options.outputfile is None:
        name, ext = inputfile.rsplit('.', 1)
        outputfile = f'{name}_{options.suffix}.{ext}'
    else:
        outputfile = options.outputfile
    logger.debug(f'Confirm output filename: {outputfile}')
    
    # check method
    _METHODS = {'bz', 'boltzmann', 
                'ss', 'smoothstep'}
    x = datas[:, 0:1]
    tc = options.t_cut
    wd = options.width / 4  # width from centre to position of 1/4 height
    method = options.method.lower()
    if method in {'bz', 'boltzmann'}:
        from .analysis import boltzmann
        # wr = boltzmann(0.25, inverse=True)
        # print(wr)
        wr = 1.0986122886681098
        factor = boltzmann((x-tc)/wd*wr)
        method = 'Boltzmann'
    elif method in {'ss', 'smoothstep'}:
        from .analysis import smoothstep
        # wr = smoothstep(0.25, inverse=True)
        # print(wr)
        wr = 0.3472963553338607
        factor = smoothstep((x-tc)/wd*wr)
        method = 'SmoothStep'
    else:
        logger.error('Failed to recognize the method for cut-off')
        logger.error(f'Now is {method}, but methods shown below are allowed: \n{_METHODS}\n')
        raise ValueError("Value of 'method' is invalid.")
    logger.info(f'Using {method} function to cut-off data')
    logger.debug(f'Transition width: {4*wd:.4f}, Tcut: {tc:.4f}')
        
    # check column
    if options.column is None:
        index = None
        logger.debug('All columns will be tailored')
    elif options.column.isdecimal():
        index = list(map(int, options.column))
        logger.info(f"Column index(es): {' '.join(s for s in options.column)}")
    else:
        index = list(map(float, options.column.split()))
        logger.info(f'Column indexes: {options.column}')
    
    if options.bare:
        datas = datas[:, index] * factor
        comment = ''
    else:
        datas[:, index] *= factor
        comment = f'Mixed datafiles - {TIME} {INFO}'
    
    # save result
    np.savetxt(outputfile, datas, fmt='%.4f', header=comment)
    logger.info(f'Save cut-off data to {outputfile}. (Done)')


def do_refine(args=None):
    import re
    
    task = 'refine'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog='')
    
    parser.add_argument('datafile', metavar='DATAFILE', nargs='+',
                        help='Filenames of datafile')
    
    parser.add_argument('-s', '--suffix', 
                        help='The suffix to generate filename of output file. \
                              If not specified, outputs are re-written datafiles')
    
    options = parser.parse_args(args)
    
    logger.info(f'{DESC} - {TIME}')
    
    suffix = options.suffix
    pattern = re.compile('[^#\n]*')
    for fn in options.datafile:
        outs = []
        with open(fn, 'r') as f:
            for line in f:
                content = pattern.match(line)
                if content and content.group():
                    outs.append(content.group()+'\n')
        if suffix:
            name, ext = fn.rsplit('.', 1)
            fn2 = f'{name}_{suffix}.{ext}'
            with open(fn2, 'w') as f:
                f.writelines(outs)
            logger.info(f'Refine {fn} ...   -> {fn2} OK')
        else:
            with open(fn, 'w') as f:
                f.writelines(outs)
            logger.info(f'Refine {fn} ... OK')
    
    logger.info('(DONE)')
