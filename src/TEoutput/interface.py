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
}

logger = get_root_logger(level=20, fmt='[%(name)s] %(message)s')


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
        logger.error('Failed to read/generate sampling points.')
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
        logger.error(f'Now is {method}, but methods shown below are allowed: \n{_METHODS}')
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

