"""Program to simulate the X-ray spectra and color of NS and CV spectra.

Functions needed:
1. Choose a random source and collect the corresponding response and background
2. A random distribution of NH and power-law for the NS spectra.
3. Random distribution of NH, power-law and Fe-line eq width for CV
4. Fake-it simulations.
Analysis should be done both for PN and MOS
"""


import copy
import glob2
import os
import xspec
import numpy as np
from astropy.io import fits


def get_resp(src_file, obs_folder='./', rmf_folder='./'):
    """Get background and responses for the given source file.

    Inputs:
    src_file = Combined source file
    obs_folder - Folder containing individual observations of the source
    """
    spectra_header = fits.open(src_file)[1].header
    bg_file = obs_folder + spectra_header['BACKFILE']
    arf_file = obs_folder + spectra_header['ANCRFILE']
    resp_file = rmf_folder + spectra_header['RESPFILE']
    exptime = spectra_header['EXPOSURE']
    file_list = [bg_file, arf_file, resp_file]
    for i, file in enumerate(file_list):
        if file.split('/')[-1] == 'None':
            file_list[i] = ''
    return file_list[0], file_list[1], file_list[2], exptime


def sim_msp(resp_file, arf_file, bg_file, exp_s, sim_msp_name, nh_val,
            gamma_val, unabs_lx_val):
    """Simulate the MSP spectra with the given values."""
    msp_settings = xspec.FakeitSettings(
        response=resp_file, arf=arf_file, background=bg_file, exposure=exp_s,
        fileName=sim_msp_name)
    msp_model = xspec.Model('tbabs*pegpwrlw')
    unabs_flux = unabs_lx_val/(7.65757E+45)
    msp_model.setPars({1: nh_val/1.0E+22, 2: gamma_val, 3: 2, 4: 10,
                       5: unabs_flux/1.0E-12})
    xspec.AllData.fakeit(1, msp_settings)
    xspec.AllData.clear()
    xspec.AllModels.clear()


def sim_msp_from_src(src_file, sim_msp_name, nh_val, gamma_val,
                     unabs_lx_val):
    """Simulate MSP spectra from a source spectra."""
    msp_settings = xspec.FakeitSettings(fileName=sim_msp_name)
    spectrum = xspec.Spectrum(src_file)
    msp_model = xspec.Model('tbabs*pegpwrlw')
    unabs_flux = unabs_lx_val/(7.65757E+45)
    msp_model.setPars({1: nh_val/1.0E+22, 2: gamma_val, 3: 2, 4: 10,
                       5: unabs_flux/1.0E-12})
    xspec.AllData.fakeit(1, msp_settings)
    xspec.AllData.clear()
    xspec.AllModels.clear()


def sim_cv_from_src(src_file, sim_msp_name, nh_val, temp_val, unabs_lx_val,
                    ew_64, ew_67, ew_70):
    """Simulate spectra of CVs."""
    ip_settings = xspec.FakeitSettings(fileName=sim_msp_name)
    spectrum = xspec.Spectrum(src_file)
    xspec.Xset.addModelString("APECNOLINES", "yes")
    ip_model = xspec.Model("tbabs*(apec+gaussian+gaussian+gaussian)")
    unabs_flux = unabs_lx_val/(7.65757E+45)
    ip_model.setPars({1: nh_val, 2: temp_val, 6: 6.4, 8: 1.0E-4, 9: 6.7,
                      11: 1.0E-4, 12: 7.0, 14: 1.0E-4})
    xspec.AllModels.eqwidth(3, rangeFrac=0.0)
    test_ew_64 = spectrum.eqwidth[0]
    xspec.AllModels.eqwidth(4, rangeFrac=0.0)
    test_ew_67 = spectrum.eqwidth[0]
    xspec.AllModels.eqwidth(5, rangeFrac=0.0)
    test_ew_70 = spectrum.eqwidth[0]
    norm_64 = ew_64/test_ew_64*1.0E-4
    norm_67 = ew_67/test_ew_67*1.0E-4
    norm_70 = ew_70/test_ew_70*1.0E-4
    xspec.AllModels.clear()
    ip_model = xspec.Model("tbabs*cflux*(apec+gaussian+gaussian+gaussian)")
    ip_model.setPars({1: nh_val, 2: 2.0, 3: 10.0, 4: np.log10(unabs_flux),
                      5: temp_val, 9: 6.4, 11: norm_64, 12: 6.7, 14: norm_67,
                      15: 7.0, 17: norm_70})
    xspec.AllData.fakeit(1, ip_settings)
    xspec.AllData.clear()
    xspec.AllModels.clear()


def msp_simulations(num_msps, nh_vals, gamma_vals, unabs_lx_vals,
                    resp_files=None, arf_files=None, bg_files=None,
                    obs_folder=None, rmf_folder=None, exp_times=None,
                    sim_msp_folder='./'):
    """Simulate MSPs."""
    if resp_files is None:
        if obs_folder is None or rmf_folder is None:
            raise ValueError('Either give the response files explicitly or' +
                             'give the observation and rmf folders from which'
                             + 'they should be sampled')
        src_files = glob2.glob(obs_folder + '*_src_grp.ds')
        src_files_forsim = np.random.choice(src_files, size=num_msps)

    for i, msp_num in enumerate(num_msps):
        if resp_files is None:
            bg_file, arf_file, resp_file, exptime = get_resp(
                src_files_forsim[i], obs_folder=obs_folder,
                rmf_folder=rmf_folder)
        else:
            bg_file = bg_files[i]
            arf_file = arf_files[i]
            resp_file = resp_files[i]
            exptime = exp_times[i]

        sim_msp(resp_file, arf_file, bg_file, exptime,
                sim_msp_folder+'msp_'+str(i)+'.fak', nh_vals[i], gamma_vals[i],
                unabs_lx_vals[i])
        if i % 1000 == 0:
            print('Finished ' + str(i) + 'simulations')


def get_xmm_src_files(src_folder):
    """Get PN and MOS src files."""
    pn_src_files = glob2.glob(src_folder + '*_PN_*src_grp1*cts.ds')
    mos_src_files = glob2.glob(src_folder + '*_MOS_*src_grp1*cts.ds')
    # pn_mos_png_files = glob2.glob(src_folder + '*_PN_MOS_combined_src.png')
    common_files_pn = []
    common_files_mos = []
    only_pn_files = copy.copy(pn_src_files)
    only_mos_files = copy.copy(mos_src_files)
    for pn_file in pn_src_files:
        pn_src_num = pn_file.split('/')[-1].split('_')[0]
        if os.path.isfile(src_folder + pn_src_num +
                          '_MOS_combined_src_grp.ds'):
            mos_file = glob2.glob(src_folder + pn_src_num +
                                  '_MOS_combined_src_grp1*cts.ds')[0]
            common_files_pn.append(pn_file)
            common_files_mos.append(mos_file)
            only_pn_files.remove(pn_file)
            only_mos_files.remove(mos_file)
    return common_files_pn, common_files_mos, only_pn_files, only_mos_files


def get_src_nums(src_file_list):
    src_nums = []
    for src_file in src_file_list:
        src_num = src_file.split('/')[-1].split('_')[0]
        src_nums.append(src_num)
    return src_nums


def msp_sims_from_src(num_msps, nh_vals, gamma_vals, unabs_lx_vals, src_folder,
                      sim_msp_folder='./', file_prefix='msp_'):
    """Simulate MSPs rom source files."""
    (common_files_pn, common_files_mos, only_pn_files,
     only_mos_files) = get_xmm_src_files(src_folder)
    src_files = common_files_pn + only_pn_files + only_mos_files
    src_args_forsim = np.random.choice(
        np.arange(len(src_files)), size=num_msps)
    for i in range(num_msps):
        if src_args_forsim[i] < len(common_files_pn) + len(only_pn_files):
            sim_msp_from_src(
                src_files[src_args_forsim[i]],
                sim_msp_folder + file_prefix + str(i) + '_PN.fak',
                nh_vals[i], gamma_vals[i], unabs_lx_vals[i])
            if src_args_forsim[i] < len(common_files_pn):
                sim_msp_from_src(
                    common_files_mos[src_args_forsim[i]],
                    sim_msp_folder + file_prefix + str(i) + '_MOS.fak',
                    nh_vals[i], gamma_vals[i], unabs_lx_vals[i])
        else:
            sim_msp_from_src(
                src_files[src_args_forsim[i]],
                sim_msp_folder + file_prefix + str(i) + '_MOS.fak',
                nh_vals[i], gamma_vals[i], unabs_lx_vals[i])

        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')


def cvs_sims_from_src(num_cvs, nh_vals, temp_vals, unabs_lx_vals, ew_64_vals,
                      ew_67_vals, ew_70_vals, src_folder, sim_cv_folder='./',
                      file_prefix='cv_'):
    """"Simulate MSPs from source files."""
    (common_files_pn, common_files_mos, only_pn_files,
     only_mos_files) = get_xmm_src_files(src_folder)
    src_files = common_files_pn + only_pn_files + only_mos_files
    src_args_forsim = np.random.choice(
        np.arange(len(src_files)), size=num_cvs)
    for i in range(num_cvs):
        if src_args_forsim[i] < len(common_files_pn) + len(only_pn_files):
            sim_cv_from_src(
                src_files[src_args_forsim[i]],
                sim_cv_folder + file_prefix + str(i) + '_PN.fak', nh_vals[i],
                temp_vals[i], unabs_lx_vals[i], ew_64_vals[i], ew_67_vals[i],
                ew_70_vals[i])
            if src_args_forsim[i] < len(common_files_pn):
                sim_cv_from_src(
                    common_files_mos[src_args_forsim[i]],
                    sim_cv_folder + file_prefix + str(i) + '_MOS.fak',
                    nh_vals[i], temp_vals[i], unabs_lx_vals[i], ew_64_vals[i],
                    ew_67_vals[i], ew_70_vals[i])
        else:
            sim_cv_from_src(
                src_files[src_args_forsim[i]],
                sim_cv_folder + file_prefix + str(i) + '_MOS.fak', nh_vals[i],
                temp_vals[i], unabs_lx_vals[i], ew_64_vals[i], ew_67_vals[i],
                ew_70_vals[i])

        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')


def get_msp_param_vals(num_msps, nh_abs_type):
    """Get parameter values for the MSP simulations."""
    if nh_abs_type == 'high':
        nh_vals = np.random.uniform(22.7, 23.7, num_msps)
    elif nh_abs_type == 'mid':
        nh_vals = np.random.uniform(22.0, 22.7, num_msps)
    elif nh_abs_type == 'low':
        nh_vals = np.random.uniform(21.0, 22.0, num_msps)
    else:
        print("'nh_abs_type' should be 'high', 'mid', or 'low'.")
    nh_vals = 10**nh_vals
    gamma_vals = np.random.uniform(1.0, 2.0, num_msps)
    lx_vals = np.random.uniform(31.0, 34.0, num_msps)
    lx_vals = 10**lx_vals
    return nh_vals, gamma_vals, lx_vals


def cv_param_vals(num_cvs, nh_abs_type, cv_type='IP'):
    """"Get parameter values for IP simulations."""
    if nh_abs_type == 'high':
        nh_vals = np.random.uniform(22.7, 23.7, num_cvs)
    elif nh_abs_type == 'mid':
        nh_vals = np.random.uniform(22.0, 22.7, num_cvs)
    elif nh_abs_type == 'low':
        nh_vals = np.random.uniform(21.0, 22.0, num_cvs)
    else:
        print("'nh_abs_type' should be 'high', 'mid', or 'low'.")
    nh_vals = 10**nh_vals
    lx_vals = np.random.uniform(31.0, 34.0, num_cvs)
    lx_vals = 10**lx_vals
    if cv_type == 'IP':
        temp_vals = np.random.normal(34.0, 4.54, num_cvs)
        ew_64_vals = np.random.normal(115.0, 9.12, num_cvs)
        ew_67_vals = np.random.normal(107, 16.0, num_cvs)
        ew_70_vals = np.random.normal(80, 6.81, num_cvs)
    elif cv_type == 'SS':
        temp_vals = np.random.normal(27.2, 20.8, num_cvs)
        ew_64_vals = np.random.normal(280, 90, num_cvs)
        ew_67_vals = np.random.normal(241, 78.3, num_cvs)
        ew_70_vals = np.random.normal(91, 20.1, num_cvs)
    else:
        print('CV types can only be IP or SS')
    return nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals, ew_70_vals


def main(num_msps=10000, nh_abs_type='high', src_folder=None,
         sim_msp_folder='./'):
    """Main function"""
    nh_vals, gamma_vals, lx_vals = get_msp_param_vals(num_msps, nh_abs_type)
    if src_folder is None:
        src_folder = './Galactic_' + nh_abs_type + 'NH_combinedXMM/'
    msp_sims_from_src(10000, nh_vals, gamma_vals, lx_vals, src_folder,
                      sim_msp_folder, 'msp_'+nh_abs_type+'NH_')
    msp_param_vals = np.column_stack(nh_vals, gamma_vals, lx_vals)
    np.savetxt(sim_msp_folder + 'paramfile.txt', msp_param_vals)
