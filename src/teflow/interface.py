#   Copyright 2023-2024 Jianbo ZHU
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
from pathlib import PurePath
from datetime import datetime

import numpy as np

from ._version import __version__
from .utils import get_pkg_name, get_root_logger

# dprint = print      # for debug using

CMD = 'tef'
CPR = 'Copyright 2023-2024 Jianbo ZHU'
PKG = get_pkg_name().replace('te', 'TE')
VISION = __version__
INFO = f'{PKG}({VISION})'
PLATFORM = f"{INFO} @ Python {sys.version}".replace('\n', '')
TIME = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
DESCRIPTION = {
    'interp': 'Data interpolation and extrapolation',
    'mixing': 'Mixing the datafile with same array-shape',
    'ztdev' : 'Calculate ZTdev of thermoelectric generator',
    'engout': 'Calculate engineering thermoelectric performance',
    'format': 'Format thermoelectric properties data',
    'cutoff': 'Cut-off data at the threshold temperature',
    'refine': 'Remove redundant & extract the concerned data',
    'band'  : 'Insight carriar transport with band models',
    'kappa' : 'Simulate & fit thermal conductivity',
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
     kappa  {DESCRIPTION['kappa']}
     ztdev  {DESCRIPTION['ztdev']}
    engout  {DESCRIPTION['engout']}
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
    # parser.add_argument('-H', '--headers', **OPTS['headers'])
    'headers': dict(
        action='store_true',
        help='Include headers without a hash character'
    ),

    # parser.add_argument('-b', '--bare', **OPTS['bare'])
    'bare': dict(
        action='store_true',
        help='Output data without header',
    ),
    
    # parser.add_argument('inputfile', **OPTS['inputf'])
    'inputf': dict(
        metavar='INPUTFILE',
        help='Input file name (must be provided)',
    ),
    
    # parser.add_argument('outputfile', **OPTS['outputf'])
    'outputf': dict(
        metavar='OUTPUTFILE', nargs='?',
        help='Output file name (optional, auto-generated if omitted)',
    ),
    
    # parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    'suffix': lambda suf: dict(
        default=f'{suf}',
        help=f'Suffix for generating the output file name (default: {suf})',
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
        elif task.startswith('engout'):
            do_engout(args[1:])
        elif task.startswith('format'):
            do_format(args[1:])
        elif task.startswith('cutoff'):
            do_cutoff(args[1:])
        elif task.startswith('refine'):
            do_refine(args[1:])
        elif task.startswith('band'):
            do_band(args[1:])
        elif task.startswith('kappa'):
            do_kappa(args[1:])
        else:
            do_help()
    else:
        print(PLATFORM)


def _wraptxt(desc, details='', indent=2, width=75):
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


def _suffixed(outputname, inputname, suffix, ext=None):
    '''
    Append suffix to inputname if outputname is absent, otherwise return itself. 
    '''
    if outputname:
        return outputname

    p = PurePath(inputname)
    if p.suffix:
        return f'{p.stem}_{suffix}{ext or p.suffix}'
    else:
        return f'{p.stem}_{suffix}'


def _to_file(options, data, header='', labels=(), fmt='%.4f', fp=None):
    if fp is None:
        fp = _suffixed(options.outputfile, options.inputfile, options.suffix)

    if isinstance(data, dict):
        labels = labels or list(data.keys())
        data = np.vstack(list(data.values())).T

    if options.bare:
        np.savetxt(fp, data, fmt=fmt)
    else:
        header = f'{header} - {TIME} {INFO}' if header else f'{TIME} {INFO}'
        labels = '  '.join(labels) if not isinstance(labels, str) else labels
        if options.headers:
            header = labels or header
            comments = ''
        else:
            header += f'\n{labels}' if labels else ''
            comments = '#'
        np.savetxt(fp, data, fmt=fmt, header=header, comments=comments)
    return fp


def do_help():
    print(INFO_HELP)
    print(FOOTNOTE)


def do_interp(args=None):
    from .analysis import interp
    
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
    
    parser.add_argument('-n', '--npoints', type=int, default=101,
        help='The number of interpolated data points (default: 101)')

    parser.add_argument('--range', metavar='START:STEP:END',
        help='Specify equal-interval points for interpolation. (e.g., 0:2:10)')

    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))

    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    x, *y = np.loadtxt(inputfile, unpack=True, ndmin=2)
    logger.info('Read data_x and data_y from {}'.format(inputfile))
    
    # parse outputfile name
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # read sampling points
    try:
        x2, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except FileNotFoundError:
        # failed to read sampling points and set them automatically
        logger.info(f'Failed to read sampling points from {outputfile}')
        
        if options.range:
            items = options.range.strip().split(':')
            if len(items) == 3:
                start, step, stop = map(float, items)
                stop += 0.001 * step
                x2 = np.arange(start, stop, step)
                logger.debug(f'Using np.arange({start}, {stop}, {step})')
            else:
                logger.debug(f'Current value of range option: {options.range}')
                raise ValueError('Failed to parse --range option: '
                                 "'START:STEP:END' format is required.")
        else:
            npoint = options.npoint
            x2 = np.linspace(x[0], x[-1], num=npoint)
            logger.debug(f'Using np.linspace({x[0]}, {x[-1]}, num={npoint}) ')
        
        logger.info('Generate sampling points automatically')
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
    from .analysis import mixing
    
    task = 'mixing'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-e', '--extend', action='store_true',
        help='Just concatenate multiple files horizontally.')

    parser.add_argument('-E','--raw-extend', action='store_true',
        help='Similar to --extend, but without any filtering.')

    parser.add_argument('-w', '--weight', metavar='WEIGHTS',
        help="Weights of mixing, with the same number of datafiles \
              (default:'1 1 1 ...')")
    
    parser.add_argument('-s', '--scale', type=float, default=0, 
        help='The scale factor (default: 1/sum(weights))')

    parser.add_argument('-o', '--outputfile', metavar='OUTPUTFILE',
        help='Specify output filename. Defaults to screen display.')

    parser.add_argument('inputfile', metavar='INPUTFILE', nargs='+',
        help='Filename of input file, which is usually more than one')

    # parser.add_argument('outputfile', metavar='OUTPUTFILE', 
    #     help='A output filename is requrired here, \
    #           which follows the inputfile(s)')

    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    filenames = options.inputfile
    outputfile = options.outputfile
    logger.info(f"Datafiles: {', '.join(filenames)}")
    logger.info(f'The output will be written to: {outputfile or "Screen"}')

    # run extend and exit, if enable
    if options.raw_extend or options.extend:
        datas = []
        for filename in filenames:
            try:
                with open(filename, 'r') as f:
                    datas.append(f.readlines())
            except IOError:
                logger.error(f'Failed to read {filename}')
                raise

        if options.raw_extend:
            logger.info('Concatenate multiple files without filtering')
            _fetch = lambda line: line.strip()
            datas = [map(_fetch, data) for data in datas]
        else:
            logger.info('Concatenate multiple files horizontally')
            _fetch = lambda line: line.split('#', 1)[0].strip()
            datas = [filter(None, map(_fetch, data)) for data in datas]

        if outputfile:
            with open(outputfile, 'w') as f:
                if not options.bare:
                    f.write(f'# Extend datafiles - {TIME} {INFO}\n')
                for lines in zip(*datas):
                    f.write(' '.join(lines) + '\n')
            logger.info(f'Save extended file to {outputfile} (Done)')
        else:
            fout = 'Result:'
            for lines in zip(*datas):
                fout += '\n  ' + ' '.join(lines)
            logger.critical(fout)
        return None

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
    if outputfile:
        comment = '' if options.bare else f'Mixed datafiles - {TIME} {INFO}'
        np.savetxt(outputfile, data, fmt='%.4f', header=comment)
        logger.info(f'Save mixed data to {outputfile} (Done)')
    else:
        fout = 'Result:'
        for line in data:
            fout += '\n  ' + ' '.join(f'{v:.4f}' for v in line)
        logger.critical(fout)


def do_ztdev(args=None):
    from .ztdev import cal_ZTdev, valuate
    
    task = 'ztdev'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=FOOTNOTE,
        description=_wraptxt(f'{DESC} - {INFO}','''
            Prepare a data file with columns in the following order:
            T, C, S, and K. Alternatively, you can format the data file
            with Tc, Th, and a pre-calculated Yita; in this case, include
            the -y (or --yita) option to indicate the use of this format.
            ''')
        )
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-y', '--yita', action='store_true',
        help='Use this option when the input file is formatted '
             'with columns Tc, Th, Yita')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
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
        # calculate ZTdev
        ZTdev = cal_ZTdev(Yita, Tc, Th)
        logger.info('Calculate ZTdev from Yita')
    else:
        data_TCSK = np.loadtxt(inputfile, unpack=True, ndmin=2)
        logger.info(f'Read data [T, C, S, K] from {inputfile}')
        result = valuate(data_TCSK[:4], allTemp=True)
        out_props = ('Tc', 'Th', 'Yita', 'ZTdev')
        Tc, Th, Yita, ZTdev = [result[k] for k in out_props]
    
    # output
    outdata = np.c_[Tc, Th, Yita, ZTdev]
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
    pp_fmt = 'Tc     Th       Yita   ZTdev'
    softinfo = f"Calculate ZTdev - {TIME} {INFO}\n{pp_fmt}"
    comment = '' if options.bare else softinfo
    np.savetxt(outputfile, outdata, fmt='%.4f', header=comment)
    logger.info(f'Save ZTdev data to {outputfile} (Done)')


def do_engout(args=None):
    from .engout import GenLeg, GenPair

    task = 'engout'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=FOOTNOTE,
        description=_wraptxt(f'{DESC} - {INFO}','''
            Prepare a data file with columns in 'TCSK' order for
            single-leg devices, or 'TCSKCSK' for paired-leg devices
            if the -p (--pair) option is enabled. The conductivity
            column can accept resistivity values using the -R
            (--resistivity) option. Currently, only evaluations
            for thermoelectric generators are supported.
            ''')
        )

    parser.add_argument('-b', '--bare', **OPTS['bare'])

    parser.add_argument('-p', '--pair', action='store_true',
        help='Enable the two-leg model, otherwise the single-leg model.')

    parser.add_argument('-R', '--resistivity', action='store_true',
        help='Use resistivity instead of the default conductivity.')

    parser.add_argument('-L', '--length', type=float, default=1,
        help='The leg length (or height) in mm. Defaults to 1 mm.')

    parser.add_argument('inputfile', **OPTS['inputf'])

    parser.add_argument('outputfile', **OPTS['outputf'])

    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))

    options = parser.parse_args(args)

    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')

    # determine paired
    paired = options.pair
    dev_type = 'unicouple' if paired else'single-leg'
    logger.debug(f'Device type: {dev_type}')

    # read datas
    inputfile = options.inputfile
    datas = np.loadtxt(inputfile, unpack=True, ndmin=2)
    if paired and (len(datas) < 7):
        raise ValueError('At least 7 columns of data are required.')
    elif len(datas) < 4:
        raise ValueError('At least 4 columns of data are required.')
    logger.info(f'Read datas from file {inputfile}.')

    # determine group
    dsp = '{} is adopted for conductive property.'
    if options.resistivity:
        datas[1] = 1E4/datas[1]
        if paired:
            datas[4] = 1E4/datas[4]
        logger.info(dsp.format('Resistivity'))
    else:
        logger.info(dsp.format('Conductivity'))

    # determine length
    length = options.length
    if length <= 0:
        raise ValueError('The length of leg must be positive.')
    logger.debug(f'Read length of leg: {length} mm')

    # calculation
    logger.info('Perform simulating of thermoelectric generator.')
    if paired:
        if datas[2].mean() < datas[5].mean():
            datas_n, datas_p = datas[:4, :], datas[[0, 4, 5, 6], :]
        else:
            datas_p, datas_n = datas[:4, :], datas[[0, 4, 5, 6], :]
        out = GenPair.valuate(datas_p, datas_n, L=length)
    else:
        out = GenLeg.valuate(datas, L=length)

    # deal with results
    out['Tc'] = 300 * np.ones_like(out['deltaT'])
    out['Th'] = 300 + out['deltaT']
    props = 'Tc     Th       PFeng  ZTeng  Pout   Yita'
    outdata = np.vstack([out[p] for p in props.split()]).T

    # output
    info = f'Engineering performance (L={length}mm, A=100mm^2)'\
           f' - {TIME} {INFO}\n{props}'
    comment = '' if options.bare else info
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
    np.savetxt(outputfile, outdata, fmt='%.4f', header=comment)
    logger.info(f'Save results to {outputfile} (Done)')


def do_format(args=None):
    from .analysis import parse_TEdatas, interp
    
    task = 'format'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=FOOTNOTE,
        description=_wraptxt(
            f'{DESC} - {INFO}','''
            Interpolate thermoelectric properties at various temperatures to
            align temperature points. This tool offers the flexibility to
            automatically generate temperature points or to explicitly define
            them in the output file. Data column attributes can be set using
            the -g/--group option. Optionally, calculations of thermoelectric
            properties can be enabled with the -c/--calculate option.
            Notably, conductivity in the output data is uniformly presented
            in S/cm. For inputs using resistivity (uOhm.m), specify this with
            'R' in the -g/--group option.
            '''
        )
    )
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-c', '--calculate', action='store_true', 
        help='Calculate thermoelectric power factor and figure-of-merit')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-m', '--method', default='cubic', 
        help="Interpolation method, only 'linear' and 'cubic' allowed "\
             '(default: cubic)')

    parser.add_argument('-D', '--temperature-step',
        type=float, default=25, metavar='TSTEP',
        help='Specify the increment (in Kelvin) for the auto temperature '\
             'series. (default: 25)')

    parser.add_argument('-g', '--group', default='TCTSTK', 
        help='Group identifiers for paired data (e.g. TCTSTK, TCSK, '\
             'TKXXTSC, default: TCTSTK)')
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
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
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # read temperatures
    try:
        T, *_ = np.loadtxt(outputfile, unpack=True, ndmin=2)
    except IOError:
        # failed to read temperatures and set them automatically
        logger.info(f'Failed to read temperatures from {outputfile}')
        
        t_step = options.temperature_step
        t_max = max(TEdatas[pp][0].max() for pp in ('C', 'S', 'K'))
        if t_step > 23:
            t_num = round((t_max-323)/t_step)+1
            T = np.array([300, ] + [323+i*t_step for i in range(t_num)])
            logger.debug(f'Temperatures: 300, 323, {323+t_step}, ..., {T[-1]}')
        else:
            t_sum = round((t_max-300)/t_step)+1
            T = np.array([300+i*t_step for i in range(t_sum)])
            logger.debug(f'Temperatures: 300, {300+t_step}, ..., {T[-1]}')
        logger.info('Generate temperatures automatically where '
                    f'Tmax = {T[-1]:g} K and Tstep = {t_step:g} K')
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
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read origin data
    inputfile = options.inputfile
    datas = np.loadtxt(inputfile, ndmin=2)
    logger.info('Read datas from {}'.format(inputfile))
    
    # parse outputfile name
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
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
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
    options = parser.parse_args(args)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read raw data and filter
    inputfile = options.inputfile
    with open(inputfile, 'r') as f:
        _fetch = lambda line: line.split('#', 1)[0].strip().split()
        contents = filter(None, map(_fetch, f.readlines()))
    logger.info(f'Clear all comments and blank lines in {inputfile}')
    
    # pick up columns
    if options.column:
        usecols = list(map(int, options.column.split()))
        _pick = lambda items: ' '.join(items[i] for i in usecols)
        logger.info(f'Column indexes which are picked up: {options.column}')
    else:
        _pick = lambda items: ' '.join(items)
    contents = map(_pick, contents)

    # parse outputfile name
    outputfile = _suffixed(options.outputfile, inputfile, options.suffix)
    logger.debug(f'Parse output filename: {outputfile}')
    
    # write data
    with open(outputfile, 'w') as f:
        if not options.bare:
            f.write(f"# Refined data - {TIME} {INFO}\n")
        for line in contents:
            f.write(line+'\n')
    logger.info(f'Save refined data to {outputfile} (Done)')


def do_band(args=None):
    from .bandlib import EXECMETA
    from .loader import TEdataset
    
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

    parser.add_argument('-H', '--headers', **OPTS['headers'])
    
    parser.add_argument('-b', '--bare', **OPTS['bare'])
    
    parser.add_argument('-g', '--group', default='STCN',
        help='Group identifiers for input data (default: STCN)')
    
    parser.add_argument('-G', '--gap', type=float, default=None,
        help='Bandgap in eV. Defaults to None, indicating the '\
             'use of a parabolic band model')
    
    parser.add_argument('-p', '--properties', default=None,
        help='Specify the properties to be considered for calculation, '\
             'separated by spaces. If not specified, '\
             'all calculated properties will be output.')
    
    parser.add_argument('inputfile', **OPTS['inputf'])
    
    parser.add_argument('outputfile', **OPTS['outputf'])
    
    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))
    
    options = parser.parse_args(args)
    # print(options)
    
    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')
    
    # read gap and determine model
    Egap = options.gap
    if Egap is None:
        model = EXECMETA['valuate.APSSPB']
        logger.info('Using single parabolic band (SPB) model.')
    elif Egap <= 0:
        model = EXECMETA['valuate.APSSPB']
        logger.warning('Faced with a negative band gap value, we are '\
            'compelled to adopt the single parabolic band (SPB) model.')
    else:
        model = EXECMETA['valuate.APSSKB']
        model.update(Eg=Egap)
        logger.info(f'Using single Kane band (SKB) model where Eg = {Egap} eV.')
    
    # read alldata and group
    inputfile = options.inputfile
    group = options.group.upper()
    alldata = TEdataset.from_file(inputfile, group=group, independent=True)
    logger.info(f'Loading data from {inputfile} with identifiers: {group}')
    
    # parse inp. and do exec.
    dataSTCN = alldata.gget('STCN', default=None)
    inp = {k:np.absolute(v) for k,v in zip('STCN', dataSTCN) if v is not None}
    logger.info('Read %s successfully', ', '.join(f'data{k}' for k in inp))
    logger.debug('Parsed data:\n%s', str(inp))
    out = model.execute(**{f'data{k}':v for k, v in inp.items()})
    
    # retain properties
    props = options.properties
    if props is not None:
        out.retain(props.strip().split(), match_order=True)
    logger.info(f'Output properties: {list(out.keys())}')

    if not options.bare:
        out.update(inp)
        for prop in reversed('STCN'):
            if prop in inp:
                out.move_to_end(prop, last=False)
    outputfile = _to_file(options, out, header='Modeling carrier transport')
    logger.info(f'Save model data to {outputfile} (Done)')


def do_kappa(args=None):
    from .kappa import parse_KappaFit

    task = 'kappa'
    DESC = DESCRIPTION[task]
    parser = argparse.ArgumentParser(
        prog=f'{CMD}-{task}',
        description=f'{DESC} - {INFO}',
        epilog=FOOTNOTE)

    parser.add_argument('-H', '--headers', **OPTS['headers'])

    parser.add_argument('-b', '--bare', **OPTS['bare'])

    parser.add_argument('-n', '--npoints',
        help='The number of predicted data points (default: 101)')

    parser.add_argument('-M', '--margin',
        help='Relative margin for extending data boundaries (default: 0.05)')

    parser.add_argument('-X', '--predict', metavar='VALUES',
        help='Specify sampling points for the prediction (default: None).')

    parser.add_argument('-T', '--temperature',
        help='The temperature for insighting phonon transport (default: 300)')

    parser.add_argument('-U', '--frequnit',
        help='Specify frequency unit of phonon (default: 2pi.THz)')

    parser.add_argument('-S', '--substituted', action='store_const', const='true',
        help='Generate a new substituted configuration file')

    parser.add_argument('-Q', '--less-output', action='store_true',
        help='Suppress detailed output files, might be handy with the Debye model.')

    parser.add_argument('inputfile', **OPTS['inputf'])

    parser.add_argument('outputfile', **OPTS['outputf'])

    parser.add_argument('-s', '--suffix', **OPTS['suffix'](task))

    options = parser.parse_args(args)
    # print(options)

    logger = get_root_logger(level=LOG_LEVEL, fmt=LOG_FMT)
    logger.info(f'{DESC} - {TIME}')

    # parse specity options
    specify = dict()
    for key in ('npoints', 'margin', 'predict', 'temperature', 'frequnit', 'substituted'):
        val = getattr(options, key)
        if val:
            specify[key] = val
    if options.less_output:
        for key in ('splitkappa', 'scattering', 'spectral', 'cumulate'):
            specify[key] = 'false'

    # parse i/o filename
    inputf = options.inputfile
    outputf = _suffixed(options.outputfile, inputf, options.suffix, '.txt')

    *_, results = parse_KappaFit(filename=inputf, specify=specify)
    xi = np.atleast_2d(results['predX'])
    yi = np.atleast_2d(results['predY'])
    labels = ['Xcolumn',] + results['predL']
    data = np.vstack([xi, yi])
    _to_file(options, data.T, 'Predict kappa', labels, fmt='%.6f', fp=outputf)
    logger.info(f'Save predict kappa to {outputf}')
    if 'rateX' in results:
        xi = np.atleast_2d(results['rateX'])
        yi = np.atleast_2d(results['rateY'])
        labels = ['freq',] + results['rateL']
        data = np.vstack([xi, yi])
        output_rate = _suffixed(None, outputf, 'rate')
        _to_file(options, data.T, 'Scattering rate', labels, fmt='%.6f', fp=output_rate)
        logger.info(f'Save scattering rate to {output_rate}')
    if 'specX' in results:
        xi = np.atleast_2d(results['specX'])
        yi = np.atleast_2d(results['specY'])
        labels = ['freq',] + results['specL']
        data = np.vstack([xi, yi])
        output_spec = _suffixed(None, outputf, 'spec')
        _to_file(options, data.T, 'Spectral kappa', labels, fmt='%.6f', fp=output_spec)
        logger.info(f'Save spectral kappa to {output_spec}')
    if 'cumuX' in results:
        xi = np.atleast_2d(results['cumuX'])
        yi = np.atleast_2d(results['cumuY'])
        labels = ['mfp',] + results['cumuL']
        data = np.vstack([xi, yi])
        output_cumu = _suffixed(None, outputf, 'cumu')
        _to_file(options, data.T, 'Cumulative kappa', labels, fmt='%.6f', fp=output_cumu)
        logger.info(f'Save cumulative kappa to {output_cumu}')
    if results.get('subs', None):
        output_subs = _suffixed(None, outputf, 'subs', '.cfg')
        with open(output_subs, 'w') as f:
            f.writelines(results['subs'])
        logger.info(f'Save new configuration file to {output_subs}')
    logger.info('(DONE)')
