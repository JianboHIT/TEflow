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
import argparse, textwrap
from datetime import datetime

from ._version import __version__
from .utils import get_pkg_name, get_root_logger

# dprint = print      # for debug using

CMD = 'tef'
CPR = 'Copyright 2023 Jianbo ZHU'
PKG = get_pkg_name().replace('te', 'TE')
VISION = __version__
INFO = f'{PKG}({VISION})'
PLATFORM = f"{INFO} @ Python {sys.version}".replace('\n', '')
TIME = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
DESCRIPTION = {
    'interp': 'Data interpolation and extrapolation',
    'mixing': 'Mixing the datafile with same array-shape',
    'ztdev' : 'Calculate ZTdev of thermoelectric generator',
    'format': 'Format thermoelectric properties data',
    'cutoff': 'Cut-off data at the threshold temperature',
    'refine': 'Remove redundant & extract the concerned data',
    'band'  : 'Insight carriar transport with band models',
}

# figlet -f 'Big Money-se' TE
# figlet -f 'Georgia11' flow
INFO_HELP = f'''
      ________  ________                                         
     |        \|        \     ,...  ,,                           
      \$$$$$$$$| $$$$$$$$   .d' ""`7MM                           
        | $$   | $$__       dM`     MM                           
        | $$   | $$  \     mMMmm    MM  ,pW"Wq.`7M'    ,A    `MF'
        | $$   | $$$$$      MM      MM 6W'   `Wb VA   ,VAA   ,V  
        | $$   | $$_____    MM      MM 8M     M8  VA ,V  VA ,V   
        | $$   | $$     \   MM      MM YA.   ,A9   VVV    VVV    
         \$$    \$$$$$$$$ .JMML.  .JMML.`Ybmd9'     W      W     
  
                 {                 f'(v{VISION}, {CPR})':>45s}
______________________________________________________________________
>>> Streamline your thermoelectric workflow from materials to devices

Usage: {CMD}-subcommand [-h] ...

Subcommands:
    format  {DESCRIPTION['format']}
      band  {DESCRIPTION['band']}
     ztdev  {DESCRIPTION['ztdev']}
    interp  {DESCRIPTION['interp']}
    mixing  {DESCRIPTION['mixing']}
    refine  {DESCRIPTION['refine']}
    cutoff  {DESCRIPTION['cutoff']}
'''
FOOTNOTE = ''

LOG_LEVEL = 20
LOG_FMT = f'[{PKG}] %(message)s'      # '[%(levelname)5s] %(message)s'
# logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)

# some public options
OPTS = {
    # parser.add_argument('-b', '--bare', **OPTS['bare'])
    'bare': dict(
        action='store_true',
        help='Output data without header',
    ),
    
    # parser.add_argument('inputfile', **OPTS['inputf'])
    'inputf': dict(
        metavar='INPUTFILE',
        help='Filename of input file (necessary)',
    ),
    
    # parser.add_argument('outputfile', **OPTS['outputf'])
    'outputf': dict(
        metavar='OUTPUTFILE', nargs='?',
        help='Filename of output file (default: Basename_suffix.Extname)',
    ),
    
    # parser.add_argument('-s', '--suffix', **OPTS['suffix']('xxxxxx'))
    'suffix': lambda suf: dict(
        default=f'{suf}',
        help=f'The suffix to generate filename of output file (default: {suf})',
    ),
}


def _do_main(args=None):
    # for test
    if args is None:
        args = sys.argv[1:]
    
    if len(args) > 0:
        global LOG_LEVEL, LOG_FMT
        LOG_LEVEL = 10
        LOG_FMT = '[%(levelname)5s] %(message)s'
        
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
        elif task.startswith('band'):
            do_band(args[1:])
        else:
            do_help()
    else:
        print(PLATFORM)


def _wraptxt(desc, details='', indent=4, width=70):
    if details:
        indentation = ' ' * indent
        contents = textwrap.fill(
            textwrap.dedent(details).strip(),
            initial_indent=indentation+'>>> ',
            subsequent_indent=indentation,
            width=width,
        )
        return desc + '\n\n' + contents
    else:
        desc


def do_help():
    print(INFO_HELP)
    print(FOOTNOTE)


def do_interp(args=None):
    import numpy as np
    from .analysis import interp
    from .utils import suffixed
    
    task = 'interp'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}', 
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
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

    parser.add_argument('-s', '--suffix', **OPTS['suffix']('interp'))

    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    x, *y = np.loadtxt(inputfile, unpack=True, ndmin=2)
    logger.info('Read data_x and data_y from {}'.format(inputfile))
    
    # parse outputfile name
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # read sampling points
    try:
        x2, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except IOError:
        # failed to read sampling points and set them automatically
        logger.info(f'Failed to read sampling points from {outputfile}')
        
        npoint = options.npoint
        x2 = np.linspace(x[0], x[-1], num=npoint)
        
        logger.info('Generate sampling points automatically')
        logger.debug(f'Using np.linspace({x[0]}, {x[-1]}, num={npoint}) ')
    except Exception as err:
        # catch other error
        logger.error('Failed to read/generate sampling points.\n')
        raise(err)
    else:
        logger.info(f'Read sampling points from {outputfile}')

    # check method
    _METHODS = {'linear', 'nearest', 'nearest-up', 
             'zero', 'slinear', 'quadratic', 'cubic', 
             'previous', 'next',
             'poly1', 'poly2','poly3','poly4','poly5',}
    method = options.method.lower()
    if method not in _METHODS:
        logger.error('Failed to recognize the method of interpolation')
        logger.error(f'Now is {method}, '
                     f'but methods shown below are allowed: \n{_METHODS}\n')
        raise ValueError("Value of 'method' is invalid.")
    
    # The strategy to extrapolate points outside the data range
    extrapolate = options.extrapolate.lower()
    toMerge = (not options.bare)
    dsp = 'Perform the interpolation operation with %s extrapolation'
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
    softinfo = f'Interpolate via {method} method - {TIME} {INFO}'
    comment = '' if options.bare else softinfo
    np.savetxt(outputfile, datas.T, fmt='%.4f', header=comment)
    logger.info(f'Save interpolation results to {outputfile} (Done)')
    

def do_mixing(args=None):
    import numpy as np
    from .analysis import mixing
    
    task = 'mixing'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-w', '--weight', metavar='WEIGHTS',
        help="Weights of mixing, with the same number of datafiles \
              (default:'1 1 1 ...')")
    
    parser.add_argument('-s', '--scale', type=float, default=0, 
        help='The scale factor (default: 1/sum(weights))')

    parser.add_argument('inputfile', metavar='INPUTFILE', nargs='+',
        help='Filename of input file, which is usually more than one')
    
    parser.add_argument('outputfile', metavar='OUTPUTFILE', 
        help='A output filename is requrired here, \
              which follows the inputfile(s)')

    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    filenames = options.inputfile
    outputfile = options.outputfile
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
    logger.info(f'Save mixed data to {outputfile} (Done)')


def do_ztdev(args=None):
    import numpy as np
    from .ztdev import optim_Yita, cal_ZTdev
    from .utils import suffixed
    
    task = 'ztdev'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-y', '--yita', action='store_true',
        help='Read the data with columns Tc, Th, Yita in order, \
              rather than the material properties T, C, S, K.')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix']('ztdev'))
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read datas
    inputfile = options.inputfile
    if options.yita:
        Tc, Th, Yita, *_ = np.loadtxt(inputfile, unpack=True, ndmin=2)
        logger.info(f'Read data [Tc, Th, Yita] from {inputfile}')
        if np.max(Yita) < 1:
            logger.warning('Detected all input Yita less than 1.'
                           'Its typical range is 0-100.')
    else:
        Th, C, S, K, *_ = np.loadtxt(inputfile, unpack=True, ndmin=2)
        Tc = Th[0]*np.ones_like(Th)
        logger.info(f'Read data [T, C, S, K] from {inputfile}')
        Yita = optim_Yita([Th, C, S, K], allTemp=True)
    
    # calculate ZTdev
    ZTdev = cal_ZTdev(Yita, Tc, Th)
    logger.info('Calculate ZTdev from Yita')
    
    # output
    outdata = np.c_[Tc, Th, Yita, ZTdev]
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    pp_fmt = 'Tc     Th       Yita   ZTdev'
    softinfo = f"Calculate ZTdev - {TIME} {INFO}\n{pp_fmt}"
    comment = '' if options.bare else softinfo
    np.savetxt(outputfile, outdata, fmt='%.4f', header=comment)
    logger.info(f'Save ZTdev data to {outputfile} (Done)')


def do_format(args=None):
    import numpy as np
    from .analysis import parse_TEdatas, interp
    from .utils import suffixed
    
    task = 'format'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-c', '--calculate', action='store_true', 
        help='Calculate thermoelectric power factor and figure-of-merit')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-m', '--method', default='cubic', 
        help='Interpolation method, only linear and cubic allowed \
              (default: cubic)')

    parser.add_argument('-g', '--group', default='TCTSTK', 
        help='Group identifiers for paired data (e.g. TCTSTK, TCSK, \
              TKXXTSC, default: TCTSTK)')
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix']('format'))
    
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
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # read temperatures
    try:
        T, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except IOError:
        # failed to read temperatures and set them automatically
        logger.info(f'Failed to read temperatures from {outputfile}')
        
        t_max = TEdatas['Tmax']
        T = np.arange(298, t_max+1, 25)
        T[0] = 300
        
        logger.info('Generate temperatures automatically where '
                    f'Tmax = {t_max} K')
        logger.debug(f'Tempeartures: 300, 323, 348, 373, ..., {t_max}')
    except Exception as err:
        # catch other error
        logger.error('Failed to read/generate temperatures.\n')
        raise(err)
    else:
        logger.info(f'Read temperatures from {outputfile}')
    
    # check method and interp
    _METHODS = {'linear', 'cubic',}
    method = options.method.lower()
    if method not in _METHODS:
        logger.error('Failed to recognize the method of interpolation')
        logger.error(f'Now is {method}, '
                     f'but methods shown below are allowed: \n{_METHODS}\n')
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
        logger.info('Calculate thermoelectric PF, ZT, etc')
    pp_fmt = ', '.join(props)
    
    # data result
    softinfo = f"Formated TE data - {TIME} {INFO}\n{pp_fmt}"
    comment = '' if options.bare else softinfo
    np.savetxt(outputfile, np.vstack(fdata).T, fmt='%.4f', header=comment)
    logger.info(f'Save formated data to {outputfile} (Done)')


def do_cutoff(args=None):
    import numpy as np
    from .utils import suffixed
    
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
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-b', '--bare', action='store_true',
        help='Output concerned columns, without header')
    
    parser.add_argument('-m', '--method', default='bz',
        help='Cut-off method (Boltzmann[bz], smoothstep[ss], default: bz)')
    
    parser.add_argument('-c', '--column', metavar='COLUMN',
        help="Indexes of columns which are tailored (default: '1 2 .. N')")
    
    parser.add_argument('-w', '--width', type=float, default=10,
        help='The transition width of cut-off function (default: 10)')
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix']('cutoff'))
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    datas = np.loadtxt(inputfile, ndmin=2)
    logger.info('Read datas from {}'.format(inputfile))
    
    # parse outputfile name
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # check method
    _METHODS = {'bz', 'boltzmann', 
                'ss', 'smoothstep'}
    x = datas[:, 0:1]
    tc = options.t_cut
    wd = options.width / 2  # half width, i.e. from 0 to bound
    method = options.method.lower()
    if method in {'bz', 'boltzmann'}:
        from .analysis import boltzmann

        factor = boltzmann(5*(x-tc)/wd)
        method = 'Boltzmann'
    elif method in {'ss', 'smoothstep'}:
        from .analysis import smoothstep

        factor = smoothstep((x-tc)/wd)
        method = 'SmoothStep'
    else:
        logger.error('Failed to recognize the method for cut-off')
        logger.error(f'Now is {method}, '
                     f'but methods shown below are allowed: \n{_METHODS}\n')
        raise ValueError("Value of 'method' is invalid.")
    logger.info(f'Using {method} function to cut-off data')
    logger.debug(f'Transition width: {2*wd:.4f}, Tcut: {tc:.4f}')

    # check column
    if options.column is None:
        index = list(range(1, datas.shape[-1]))
        logger.debug('All columns will be tailored')
    else:
        index = list(map(int, options.column.split()))
        logger.info(f'Column indexes: {options.column}')
    
    if options.bare:
        datas = datas[:, index] * factor
        comment = ''
    else:
        datas[:, index] *= factor
        comment = f'Cut-off data at {tc:.2f} - {TIME} {INFO}'
    
    # save result
    np.savetxt(outputfile, datas, fmt='%.4f', header=comment)
    logger.info(f'Save cut-off data to {outputfile} (Done)')


def do_refine(args=None):
    from .utils import suffixed, purify

    task = 'refine'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-c', '--column', metavar='COLUMN',
        help="Indexes of columns which are picked up (default: '0 1 2 .. N')")
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix']('refine'))
    
    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read raw data and filter
    inputfile = options.inputfile
    with open(inputfile, 'r') as f:
        # check columns
        if options.column:
            index = list(map(int, options.column.split()))
            contents = purify(f.readlines(), usecols=index)
            logger.info(f'Column indexes which are picked up: {options.column}')
        else:
            contents = purify(f.readlines())
    logger.info(f'Clear all comments and blank lines in {inputfile}')
    
    # parse outputfile name
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # write data
    with open(outputfile, 'w') as f:
        if not options.bare:
            f.write(f"# Refined data - {TIME} {INFO}\n")
        for line in contents:
            f.write(line+'\n')
    logger.info(f'Save refined data to {outputfile} (Done)')


def do_band(args=None):
    import numpy as np
    from .bandlib import APSSPB, APSSKB, q
    from .utils import suffixed
    
    task = 'band'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=FOOTNOTE,
        description=_wraptxt(f'{DESC} - {INFO}','''
            Ensure your data file is formatted with columns for the Seebeck
            coefficient, and optionally, temperature, conductivity, and
            carrier concentration. Alter this arrangement with the -g(--group)
            option. Anticipate outputs like the Lorenz number,
            temperature-independent weighted mobility, effective mass, etc.,
            based on your supplied data.
            ''')
        )
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-g', '--group', default='STCN',
        help='Group identifiers for paired data (default: STCN)')
    
    parser.add_argument('--gap', type=float, default=None,
        help='Bandgap in eV. Defaults to None, indicating the '\
             'use of a parabolic band model')
    
    parser.add_argument('-p', '--properties', default=None,
        help='Specify theproperties to be considered for calculation, '\
             'separated by spaces. If not specified, '\
             'all calculated properties will be output.')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read gap
    Egap = options.gap
    if Egap is None:
        logger.info('Using single parabolic band (SPB) model.')
    elif Egap <= 0:
        logger.warning('Faced with a negative band gap value, we are '\
            'compelled to adopt the single parabolic band (SPB) model.')
        Egap = None
    else:
        logger.info(f'Using single Kane band (SKB) model where Eg = {Egap} eV.')
    
    # read alldata
    inputfile = options.inputfile
    alldata = np.loadtxt(inputfile, unpack=True, ndmin=2)
    logger.info(f'Load data from {inputfile} with {len(alldata)} columns')
    
    inp = dict()
    
    # parse
    group = options.group.upper()
    Ndata = len(alldata)
    if Ndata < len(group):
        group = group[:Ndata]
    if 'S' in group:
        dataS = inp['S'] = alldata[group.index('S')]
        logger.info('Fetch Seebeck coefficients successfully')
    else:
        logger.info('Failed to fetch Seebeck coefficients')
        raise IOError('Seebeck coefficients are required')
    
    if 'T' in group:
        dataT = inp['T'] = alldata[group.index('T')]
        logger.info('Fetch temperature points successfully')
    else:
        logger.info('Failed to fetch temperature points')
        if Egap is None:
            dataT = None
        else:
            raise IOError('Temperatures are required for Kane model')

    if 'C' in group:
        dataC = inp['C'] = alldata[group.index('C')]
        logger.info('Fetch electrical conductivity successfully')
    elif 'R' in group:
        dataC = inp['C'] = 1E4 / alldata[group.index('R')]
        logger.info('Fetch electrical resistivity successfully')
    else:
        dataC = None
        logger.info('Failed to fetch electrical conductivity')
    
    if 'N' in group:
        dataN = inp['N'] = alldata[group.index('N')]
        logger.info('Fetch carrier concentration successfully')
    else:
        dataN = None
        logger.info('Failed to fetch carrier concentration')

    if 'U' in group:
        dataU = alldata[group.index('U')]
        if (dataC is None) and (dataN is not None):
            dataC = inp['C'] = dataN*1E19 * q * dataU
            logger.info('Calculate electrical conductivity by N and U.')
        elif (dataN is None) and (dataC is not None):
            dataN = inp['N'] = dataC /(dataU * q * 1E19)
            logger.info('Calculate carrier concentration by C and U.')
        else:
            logger.info('Carrier mobility data is useless here.')
    
    # valuate
    if Egap is None:
        out = APSSPB.valuate(dataS, dataT, dataC, dataN)
    else:
        out = APSSKB.valuate(dataS, dataT, dataC, dataN, Eg=Egap)
    
    # retain properties
    props = options.properties
    if props is not None:
        out.retain(props.strip().split(), match_order=True)
    logger.info(f'Output properties: {list(out.keys())}')

    if options.bare:
        comment = ''
    else:
        out.update(inp)
        for prop in reversed('STCN'):
            if prop in inp:
                out.move_to_end(prop, last=False)
        comment = f"Modeling carrier transport - {TIME} {INFO}\n"\
                  f"{'  '.join(out.keys())}"
    outdata = np.vstack(list(out.values())).T
    
    outputfile = suffixed(options.outputfile, inputfile, options.suffix)
    np.savetxt(outputfile, outdata, fmt='%.4f', header=comment)
    logger.info(f'Save model data to {outputfile} (Done)')
