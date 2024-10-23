'''Program to combine the EPIC PN and MOS spectra separately.'''


import os
import subprocess
import glob2
import cycler
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import xspec


def plot_spectra(xdatas, ydatas, xerrs, yerrs, labels=None, savefigname=None):
    """Plot spectra."""
    plt.rcParams["figure.figsize"] = (6.4, 4.8)
    plt.rcParams["axes.titlesize"] = 16.0
    plt.rcParams["axes.labelsize"] = 12.0
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markersize"] = 1.0
    plt.rcParams["xtick.labelsize"] = 12.0
    plt.rcParams["ytick.labelsize"] = 12.0
    plt.rcParams["legend.fontsize"] = 16.0
    plt.rcParams["axes.prop_cycle"] = cycler.cycler(
        color=['#0f2080', '#a95aa1', '#f5793a', '#85c0f9'])
    markers = ['o', 'x', 'd', 's']
    plt.figure()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Normalized counts s$^{-1}$ keV$^{-1}$')
    plt.xscale('log')
    # Plot for multiple data groups
    if isinstance(ydatas[0], (list, np.ndarray)):
        if labels is None:
            labels = [None]*len(xdatas)
        for i, xdata in enumerate(xdatas):
            plt.errorbar(xdata, ydatas[i], yerr=yerrs[i], xerr=xerrs[i],
                         marker=markers[i], linestyle='', label=labels[i])
    else:    # Plot for a single data group
        plt.errorbar(xdatas, ydatas, yerr=yerrs, xerr=xerrs,
                     marker=markers[0], linestyle='', label=labels)

    plt.legend()
    plt.tight_layout()
    if savefigname:
        plt.savefig(savefigname)
    plt.close()


def combine_stringlist(string_array):
    """Combine the strings in an array into one string separated by spaces."""
    full_string = ""
    for string in string_array:
        full_string += string + " "
    full_string = full_string[:-1]
    return full_string


def get_bg_arf_resp(spectra, bg_directory='./', arf_directory='./',
                    resp_directory='./',
                    resp_dir2=None):
    """Get the background, ARF, and response files."""
    spectra_header = fits.open(spectra)[1].header
    print(spectra_header['NAXIS2'])
    if spectra_header['NAXIS2'] == 800:
        if resp_dir2 is None:
            resp_dir2 = (
                '/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                'data/XMM_responses/MOS_15eV/')
    #    resp_directory = resp_dir2
        return '15eVbin', '15eVbin', '15eVbin'
    bg_file = bg_directory + spectra_header['BACKFILE']
    arf_file = arf_directory + spectra_header['ANCRFILE']
    resp_file = resp_directory+spectra_header['RESPFILE']
    if not os.path.isfile(bg_file):
        raise OSError('Background spectra file ' + bg_file + ' does not exist')

    if 'EXPOSURE' not in fits.open(bg_file)[1].header:
        print(spectra, bg_file)
        bg_file = 'Invalid'

    if not os.path.isfile(arf_file):
        raise OSError('Ancillary response file ' + arf_file +
                      ' does not exist')
    if not os.path.isfile(resp_file):
        print('Adding .gz to response file name')
        resp_file = resp_file + '.gz'
        if not os.path.isfile(resp_file):
            print('Response file ' + resp_file + ' does not exist.' +
                  ' "_v20.0" is appended to the response file name.')
            resp_file = resp_file[:-7] + '_v20.0.rmf'
            if not os.path.isfile(resp_file):
                raise OSError('Response matrix file ' + resp_file +
                              ' does not exist')
    return bg_file, arf_file, resp_file


def get_combine_det_inputs(source_num, detector, src_dir='./', rmf_dir='./'):
    """Get src & bg spectra & resp files for given source and detector."""
    det_folder = src_dir+source_num + '/EPIC_' + detector + '_spec'
    if detector in ('MOS1', 'MOS2'):
        detector = 'MOS_5eV'
    # Check if the source has EPIC spectra in detector
    if os.path.isdir(det_folder):
        src_specs = glob2.glob(det_folder + '/*SRSPEC*.FTZ')
        if len(src_specs) == 0:
            return None
        print(len(src_specs), src_specs)
        src_specs_refined = []
        bg_specs = []
        arf_files = []
        rmf_files = []
        # invalid_exp_srcfiles = []
        # srcfiles_15evbins = []
        for spec in src_specs:
            bg_file, arf_file, rmf_file = get_bg_arf_resp(
                spec, bg_directory=det_folder+'/',
                arf_directory=det_folder+'/',
                resp_directory=rmf_dir+detector+'/')
            print(bg_file, spec)
            if bg_file not in ('Invalid', '15eVbin'):
                src_specs_refined.append(spec)
                bg_specs.append(bg_file)
                arf_files.append(arf_file)
                rmf_files.append(rmf_file)
            else:
                print(bg_file)

        print(len(src_specs_refined), len(bg_specs), len(arf_files),
              len(rmf_files))

        return (combine_stringlist(src_specs_refined),
                combine_stringlist(bg_specs),
                combine_stringlist(arf_files),
                combine_stringlist(rmf_files))
    return None


def get_combinepn_inputs(source_num, src_dir='./', rmf_dir='./'):
    """Get EPIC-PN source spectra, background and responses."""
    return get_combine_det_inputs(source_num, 'PN', src_dir, rmf_dir)


def get_combinemos_inputs(source_num, src_dir='./', rmf_dir='./'):
    """Get EPIC-MOS1 and EPIC-MOS2 spectra, background and responses.

    Here we combine the spectra of mos1 and mos2.
    """
    mos1_spec_strings = get_combine_det_inputs(source_num, 'MOS1', src_dir,
                                               rmf_dir)
    mos2_spec_strings = get_combine_det_inputs(source_num, 'MOS2', src_dir,
                                               rmf_dir)
    if mos1_spec_strings is not None and mos2_spec_strings is not None:
        combined_mos_spec_strings = []
        for i, string in enumerate(mos1_spec_strings):
            combined_mos_spec_strings.append(
                combine_stringlist([string, mos2_spec_strings[i]]))
        return combined_mos_spec_strings
    if mos1_spec_strings is not None:
        return mos1_spec_strings
    if mos2_spec_strings is not None:
        return mos2_spec_strings
    return None


def get_totalcounts(spectra):
    """Get the total number of counts in the given spectra."""
    spec_counts = fits.getdata(spectra)['counts']
    return np.sum(spec_counts)


def change_header_intrume(fits_file, updated_intrume):
    """Manually change the value of keyword 'INSTRUME' in the header."""
    fits_hdutables = fits.open(fits_file, mode='update')
    orig_instrume = fits_hdutables[0].header['INSTRUME']
    fits_hdutables[0].header['INSTRUME'] = updated_intrume
    fits_hdutables[0].header['HISTORY'] = (
        'INSTRUME value changed from ' + orig_instrume + ' to ' +
        updated_intrume)
    fits_hdutables.flush()
    fits_hdutables.close()


def change_header_back_respfile(fits_file):
    """"Change the response and background file."""
    fits_hdutables = fits.open(fits_file, mode='update')
    filename = os.path.basename(fits_file)
    src_num, det = filename.split('_')[:2]
    resp_file = src_num + '_' + det + '_combined_rsp_grp.ds'
    back_file = src_num + '_' + det + '_combined_bkg_grp.ds'
    fits_hdutables[1].header['RESPFILE'] = resp_file
    fits_hdutables[1].header['BACKFILE'] = back_file
    fits_hdutables.flush()
    fits_hdutables.close()


def insert_header_key(fits_file, keyword, key_value):
    """Insert keyword."""
    fits_hdutables = fits.open(fits_file, mode='update')
    fits_hdutables[1].header[keyword] = key_value
    fits_hdutables[1].header['HISTORY'] = (
        'Added the keyword ' + keyword + ' with value ' + str(key_value)
    )
    fits_hdutables.flush()
    fits_hdutables.close()


def merge_xmmspec(src_spec_str, bkg_spec_str, arf_spec_str,
                  rmf_spec_str, src_num=None, det=None, outputdir='./'):
    """Merge XMM spectra."""
    # Get source number and detector name if not given
    if src_spec_str in ('', ' '):
        print('Hello')
        return 0

    print('String = ', src_spec_str)
    print('Length = ', len(src_spec_str))
    if src_num is None or det is None:
        src_spec0_split = src_spec_str.split(' ')
        if src_num is None:
            src_num = src_spec0_split[-3]
        if det is None:
            det = src_spec0_split[-2][5:-5]

    print('Merging ' + det + ' spectra for source ' + src_num)

    # Merge spectra. Note that epicspeccombine groups the spectra by default
    output_base_str = outputdir + src_num + '_' + det + '_combined_'

    # if glob2.glob(output_base_str+'src_grp1_*cts.ds') != []:
    #     print('Output grouped spectra already exists')
    #     return 0
    # print(glob2.glob(output_base_str+'src_grp1_*cts.ds'))

    subprocess.run(['epicspeccombine', 'pha='+src_spec_str,
                    'bkg='+bkg_spec_str, 'rmf='+rmf_spec_str,
                    'arf='+arf_spec_str,
                    'filepha='+output_base_str+'src_grp.ds',
                    'filebkg='+output_base_str+'bkg_grp.ds',
                    'filersp='+output_base_str+'rsp_grp.ds',
                    'allowHEdiff=yes'], check=False)

    # Get the total counts in the spectra to finally rename the grouped spectra
    # spec_totcount = get_totalcounts(output_base_str+'src_grp.ds')

    # Change the instrument name. The epicspeccombine gives some problem, yet
    # to figure out.
    if det == 'PN':
        change_header_intrume(output_base_str+'src_grp.ds', 'EPN     ')
        change_header_intrume(output_base_str+'bkg_grp.ds', 'EPN     ')
    else:
        change_header_intrume(output_base_str+'src_grp.ds', 'EMOS1   ')
        change_header_intrume(output_base_str+'src_grp.ds', 'EMOS1   ')

    # Regroup spectra such that each spectra has at least 1 count per bin.
    subprocess.run([
        "specgroup", "spectrumset="+output_base_str+"src_grp.ds",
        "setbad=0:0.2, 10.0-15.0", "units=KEV", "mincounts=1",
        "rmfset="+output_base_str+'rsp_grp.ds',
        "groupedset="+output_base_str+'grp1_src.ds'],
                   check=False)
    return 1


def plot_source(specfile, specfile2=None, plot_device_set=False, labels=None,
                figname=None):
    """Plot xspec spectra."""
    if plot_device_set:
        xspec.Plot.device = '/xw'
        xspec.Plot.xAxis = 'keV'
        xspec.Plot.setRebin(3, 10, 1)
        xspec.Plot.setRebin(3, 10, 2)
    spec = xspec.Spectrum(specfile)
    spec.ignore('0.0-2.0, 10.0-**')
    if specfile2 is not None:
        spec2 = xspec.Spectrum(specfile2)
        spec2.ignore('0.0-0.2, 10.0-**')
    xspec.Plot('data')
    spec_x = xspec.Plot.x()
    spec_xerr = xspec.Plot.xErr()
    spec_y = xspec.Plot.y()
    spec_yerr = xspec.Plot.yErr()
    if specfile2 is not None:
        spec2_x = xspec.Plot.x(2)
        spec2_xerr = xspec.Plot.xErr(2)
        spec2_y = xspec.Plot.y(2)
        spec2_yerr = xspec.Plot.yErr(2)
        plot_spectra([spec_x, spec2_x], [spec_y, spec2_y],
                     [spec_xerr, spec2_xerr], [spec_yerr, spec2_yerr],
                     labels, figname)
    else:
        plot_spectra(spec_x, spec_y, spec_xerr, spec_yerr, labels, figname)

    xspec.AllData.clear()


def merge_source(source_num, src_dir='./', rmf_dir='./', output_dir='./'):
    """Merge PN, and MOS spectra for the source_num."""
    output_pn_spec = output_dir + source_num + '_PN_combined_src_grp.ds'
    output_mos_spec = output_dir + source_num + '_MOS_combined_src_grp.ds'
    if os.path.exists(output_pn_spec):
        combined_pn_strings = None
    else:
        combined_pn_strings = get_combinepn_inputs(source_num, src_dir,
                                                   rmf_dir)
    if os.path.exists(output_mos_spec):
        combined_mos_strings = None
    else:
        combined_mos_strings = get_combinemos_inputs(source_num, src_dir,
                                                     rmf_dir)
    if combined_pn_strings is not None:
        merge_xmmspec(combined_pn_strings[0], combined_pn_strings[1],
                      combined_pn_strings[2], combined_pn_strings[3],
                      src_num=source_num, det='PN', outputdir=output_dir)
    if combined_mos_strings is not None:
        merge_xmmspec(combined_mos_strings[0], combined_mos_strings[1],
                      combined_mos_strings[2], combined_mos_strings[3],
                      src_num=source_num, det='MOS', outputdir=output_dir)


def plot_folder(folder_name, output_dir='./', source_nums=None, plot=True):
    """Plot all spectra in a given folder."""
    if folder_name[-1] != '/':
        folder_name += '/'
    if output_dir[-1] != '/':
        output_dir += '/'

    if source_nums is None:
        specfiles = glob2.glob(folder_name + '*src_grp1*.ds')
        source_nums_prilim = []
        for spec in specfiles:
            filename = spec.split('/')[-1]
            source_num = filename[:filename.find('_')]
            source_nums_prilim.append(source_num)
        source_nums = list(set(source_nums_prilim))
    if not plot:
        return source_nums

    for i, source_num in enumerate(source_nums):
        plot_device_set = not bool(i)
        pn_spec = glob2.glob(
            folder_name + source_num + '_PN_combined_src_grp1_*.ds')
        mos_spec = glob2.glob(
            folder_name + source_num + '_MOS_combined_src_grp1_*.ds')
        if len(pn_spec) == 1 and len(mos_spec) == 1:
            figname = output_dir + source_num + '_PN_MOS_combined_src2.png'
            plot_source(pn_spec[0], specfile2=mos_spec[0],
                        plot_device_set=plot_device_set, labels=['PN', 'MOS'],
                        figname=figname)
        elif len(pn_spec) == 1 and len(mos_spec) == 0:
            figname = output_dir + source_num + '_PN_combined_src.png'
            plot_source(pn_spec[0],
                        plot_device_set=plot_device_set, labels='PN',
                        figname=figname)
        elif len(pn_spec) == 0 and len(mos_spec) == 1:
            figname = output_dir + source_num + '_MOS_combined_src.png'
            plot_source(mos_spec[0],
                        plot_device_set=plot_device_set, labels='MOS',
                        figname=figname)
        else:
            print(pn_spec)
            print(mos_spec)
            raise IOError('Weird combination. pn_spec has length ' +
                          str(len(pn_spec)) + ' and mos_spec has length ' +
                          str(len(mos_spec)))

    return source_nums


def main(src_dir, outputdir='./', rmf_dir='./'):
    """Main file. Apply merge_source to all sources in the source directory."""
    if src_dir[-1] != '/':
        src_dir += '/'
    for source_num in glob2.glob(src_dir + '*/'):
        source_num = source_num.split('/')[-2]
        merge_source(source_num, src_dir, rmf_dir, outputdir)
