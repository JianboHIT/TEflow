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

import sys
import argparse
from datetime import datetime

from ._version import __version__
from .utils import get_pkg_name, get_root_logger

CMD = 'teop'
CPR = 'Copyright 2023 Jianbo ZHU'
PKG = get_pkg_name()
VISION = __version__
INFO = f'{PKG}({VISION})'
TIME = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
DESCRIPTION = {
    'interp': 'Data interpolation and extrapolation',
    'mixing': 'Mixing the datafile with same array-shape',
    'ztdev' : 'Calculate ZTdev of thermoelectric generator',
    'format': 'Format thermoelectric properties data',
    'cutoff': 'Cut-off data at the threshold temperature',
    'refine': 'Remove all comments and blank lines in file',
}
DESCRIPTION_FMT = '\n'.join('{:>10s}    {}'.format(key, value) 
                            for key, value in DESCRIPTION.items())
# figlet -f slant TEoutput | boxes -d stark1
INFO_HELP = f"""\
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!     ____________            __              __    !
!    /_  __/ ____/___  __  __/ /_____  __  __/ /_   !
!     / / / __/ / __ \/ / / / __/ __ \/ / / / __/   !
!    / / / /___/ /_/ / /_/ / /_/ /_/ / /_/ / /_     !
!   /_/ /_____/\____/\__,_/\__/ .___/\__,_/\__/     !
!                            /_/                    !
!   {                 f'(v{VISION}, {CPR})':>45s}   !
!                                                   !
! ** A python3 package for thermoelectric output ** !
! ********** performance calculations. ************ !
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Usage: {CMD}-xxxxxx ...

Subcommands:
{DESCRIPTION_FMT}
"""
FOOTNOTE = ''

LOG_LEVEL = 20
LOG_FMT = '[%(name)s] %(message)s'      # '[%(levelname)5s] %(message)s'
# logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)


def do_main(args=None):
    # for test
    if args is None:
        args = sys.argv[1:]
    
    if len(args) > 0:
        task = args[0].lower()
        if task.startswith('interp'):
            do_interp(args[1:])
        elif task.startswith('mixing'):
            do_mixing(args[1:])
        elif task.startswith('ztdev'):
            do_ztdev(args[1:])
        elif task.startswith('format'):
            do_format(args[1:])
        elif task.startswith('cutoff'):
            do_cutoff(args[1:])
        elif task.startswith('refine'):
            do_refine(args[1:])
        else:
            print(INFO_HELP)
            print(FOOTNOTE)
    else:
        print(f"{INFO} @ Python {sys.version}".replace('\n', ''))


def do_interp(args=None):
    import numpy as np
    from .analysis import interp
    
    task = 'interp'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}', 
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
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

    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
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
        epilog=FOOTNOTE)
    
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
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
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
        epilog=FOOTNOTE)
    
    # parser.add_argument('yita', metavar='Yita', type=float,
    #                     help='Efficiency of thermoelectric generator(0 - 100)')
    
    parser.add_argument('-y', '--yita', metavar='EFFICIENCY', type=float, 
                        help='Efficiency of thermoelectric generator (0-100)')
    
    parser.add_argument('tmin', type=float, 
                        help='Temperature at cold side in K')
    
    parser.add_argument('tmax', type=float,
                        help='Temperature at hot side in K')
    
    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    yita = options.yita
    tmin = options.tmin
    tmax = options.tmax
    logger.info(f'Yita: {yita:.2f} %,  Tmin: {tmin:.2f} K, Tmax: {tmax:.2f} K')
    
    ZTdev = cal_ZTdev(yita, Tc=tmin, Th=tmax)
    logger.info(f'ZTdev: {ZTdev:.4f}. (DONE)')


def do_format(args=None):
    import numpy as np
    from .analysis import parse_TEdatas, interp
    
    task = 'format'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', action='store_true',
                        help='Output data without header')
    
    parser.add_argument('-c', '--calculate', action='store_true', 
                        help='Calculate thermoelectric power factor and figure-of-merit')
    
    parser.add_argument('inputfile', metavar='INPUTFILE',
                        help='Filename of input file (necessary)')
    
    parser.add_argument('outputfile', metavar='OUTPUTFILE', nargs='?',
                        help='Filename of output file (default: Basename_suffix.Extname)')
    
    parser.add_argument('-m', '--method', default='cubic', 
                        help='Interpolation method, only linear and cubic allowed \
                             (default: cubic)')

    parser.add_argument('-g', '--group', default='TCTSTK', 
                        help='Group identifiers for paired data (e.g. TCTSTK, TCSK, TKXXTSC, \
                             default: TCTSTK)')
    
    parser.add_argument('-s', '--suffix', default='format', 
                        help='The suffix to generate filename of output file (default: format)')
    
    options = parser.parse_args(args)
    # print(options)
    
    # logger = get_root_logger(level=10, fmt=LOG_FMT)
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    _SEPS = [('whitespace', None),
             ('comma', ','),
             ('tab', '\t')]
    inputfile = options.inputfile
    for name, sep in _SEPS:
        try:
            logger.debug(f'Using {name} as delimiter to read data')
            datas = np.genfromtxt(inputfile, delimiter=sep, unpack=True, ndmin=2)
            if np.all(np.isnan(datas)):
                raise ValueError('Failed to split up the data')
        except Exception:
            logger.debug(f"Failed to read data from file")
        else:
            logger.info(f"Read data from {inputfile} successfully")
            datas_fmt = np.array_str(datas.T, max_line_width=78, precision=2)
            logger.debug(f"Data details: \n{datas_fmt}")
            break
    else:
        raise IOError(f'Failed to read datas from {inputfile}')
    
    # parse TEdatas
    group = options.group
    TEdatas = parse_TEdatas(datas=datas, group=group)
    logger.info(f"Column identifiers: {', '.join(group)}")
    logger.info('Parse thermoelectric properties and corresponding temperatures')
    
    # parse outputfile name
    if options.outputfile is None:
        name, ext = inputfile.rsplit('.', 1)
        outputfile = f'{name}_{options.suffix}.{ext}'
    else:
        outputfile = options.outputfile
    
    # read temperatures
    try:
        T, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except IOError:
        # failed to read temperatures and set them automatically
        logger.info(f'Failed to read temperatures from {outputfile}.')
        
        t_max = TEdatas['Tmax']
        T = np.arange(298, t_max+1, 25)
        T[0] = 300
        
        logger.info(f'Generate temperatures automatically where Tmax = {t_max} K.')
        logger.debug(f'Tempeartures: 300, 323, 348, 373, ..., {t_max}')
    except Exception as err:
        # catch other error
        logger.error('Failed to read/generate temperatures.\n')
        raise(err)
    else:
        logger.info(f'Read temperatures from {outputfile}.')
    
    # check method and interp
    _METHODS = {'linear', 'cubic',}
    method = options.method.lower()
    if method not in _METHODS:
        logger.error('Failed to recognize the method of interpolation.')
        logger.error(f'Now is {method}, but methods shown below are allowed: \n{_METHODS}\n')
        raise ValueError("Value of 'method' is invalid.")
    fdata = [T, ]
    for pp in ('C', 'S', 'K'):
        fdata.append(interp(*TEdatas[pp], T, method=method))
    
    # calculate PF, ZT and PF
    props = ['T', 'C', 'S', 'K']
    if options.calculate:
        from scipy.integrate import cumtrapz
        
        props.extend(['PF', 'ZT', 'ZTave', 'CF'])
        # props.extend(['PF', 'ZT', 'ZTave_H', 'ZTave_C', 'CF'])
        
        fdata.append(fdata[1]*fdata[2]*fdata[2]*1E-6) # PF
        fdata.append(fdata[4]/fdata[3]*fdata[0]*1E-4) # ZT
        RTh = cumtrapz(1E4/fdata[1], fdata[0], initial=0)
        STh = cumtrapz(fdata[2], fdata[0], initial=0)
        KTh = cumtrapz(fdata[3], fdata[0], initial=0)
        TTh = (fdata[0][0]+fdata[0])/2
        ZTave_H = np.divide(1E-6*np.power(STh, 2)*TTh, RTh*KTh, 
                            out=1.0*fdata[5],
                            where=(np.abs(KTh)>1E-3))
        fdata.append(ZTave_H)
        # RTc = RTh[-1]-RTh
        # STc = STh[-1]-STh
        # KTc = KTh[-1]-KTh
        # TTc = (fdata[0]+fdata[0][-1])/2
        # ZTave_C = np.divide(1E-6*np.power(STc, 2)*TTc, RTc*KTc, 
        #                     out=1.0*fdata[5],
        #                     where=(np.abs(KTc)>1E-3))
        # fdata.append(ZTave_C)
        fdata.append(1E6*(np.sqrt(1+fdata[5])-1)/(fdata[0]*fdata[2]))
        logger.info('Calculate thermoelectric PF, ZT, etc.')
    pp_fmt = ', '.join(props)
    
    # data result
    comment = '' if options.bare else f"Formated TE data - {TIME} {INFO}\n{pp_fmt}"
    np.savetxt(outputfile, np.vstack(fdata).T, fmt='%.4f', header=comment)
    logger.info(f'Save formated data to {outputfile}. (Done)')


def do_cutoff(args=None):
    import numpy as np
    
    # >>>>> import when necessary <<<<<<
    # from .analysis import boltzmann
    # from .analysis import smoothstep
    
    task = 'cutoff'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
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
    
    parser.add_argument('-w', '--width', type=float, default=10,
                        help='The transition width of cut-off function (default: 10)')
    
    parser.add_argument('-s', '--suffix', default='cutoff', 
                        help='The suffix to generate filename of output file (default: cutoff)')
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
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
    else:
        index = list(map(int, options.column.split()))
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
        epilog=FOOTNOTE)
    
    parser.add_argument('datafile', metavar='DATAFILE', nargs='+',
                        help="Filenames of datafile.")
    
    parser.add_argument('-s', '--suffix', 
                        help='The suffix to generate filename of output file. \
                              If not specified, outputs are re-written datafiles')
    
    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    logger.debug(r"Regex sub: '[,\s]*#.*$' -> '', '^[,\s]*\n' -> ''")
    logger.debug(r"(Ref `sed 's/[ ,\t]*#.*$//; /^[ ,\t]*$/d' DATAFILE`)")
    
    rm_cmt = True   # remove comments
    rm_blk = True   # remove blank lines
    suffix = options.suffix
    pattern_cmt = re.compile(r'[,\s]*#.*$', flags=re.M)
    pattern_blk = re.compile(r'^[,\s]*\n', flags=re.M)
    for fn in options.datafile:
        with open(fn, 'r') as f:
            out = f.read()
            if rm_cmt:
                out = pattern_cmt.sub('', out)
            if rm_blk:
                out = pattern_blk.sub('', out)
        if suffix:
            name, ext = fn.rsplit('.', 1)
            fn2 = f'{name}_{suffix}.{ext}'
            with open(fn2, 'w') as f:
                f.write(out)
            logger.info(f'Refine {fn} ...   -> {fn2}  OK')
        else:
            with open(fn, 'w') as f:
                f.write(out)
            logger.info(f'Refine {fn} ... OK')
    
    logger.info('(DONE)')
