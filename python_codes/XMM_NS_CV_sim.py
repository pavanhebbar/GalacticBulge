"""Program to simulate the X-ray spectra and color of NS and CV spectra.

Functions needed:
1. Choose a random source and collect the corresponding response and background
2. A random distribution of NH and power-law for the NS spectra.
3. Random distribution of NH, power-law and Fe-line eq width for CV
4. Fake-it simulations.
Analysis should be done both for PN and MOS
Use functions with '_fromsrc' to generate spectra with background.
"""


from calendar import c
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
    """Simulate the MSP spectra with the given values.

    CANNOT account the background properly
    """
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
    """Simulate MSP spectra from a source spectra.

    Cannot account for the background.
    """
    msp_settings = xspec.FakeitSettings(fileName=sim_msp_name)
    spectrum = xspec.Spectrum(src_file)
    msp_model = xspec.Model('tbabs*pegpwrlw')
    unabs_flux = unabs_lx_val/(7.65757E+45)
    msp_model.setPars({1: nh_val/1.0E+22, 2: gamma_val, 3: 2, 4: 10,
                       5: unabs_flux/1.0E-12})
    xspec.AllData.fakeit(1, msp_settings)
    xspec.AllData.clear()
    xspec.AllModels.clear()


def sim_cv_from_src(src_file, sim_cv_name, nh_val, temp_val, unabs_lx_val,
                    ew_64, ew_67, ew_70):
    """Simulate spectra of CVs.
    
    Use this for simulations that need background. The background from the
    src_file will be used for the fake spectra
    """
    # Input file name of fake spectra
    ip_settings = xspec.FakeitSettings(fileName=sim_cv_name)
    spectrum = xspec.Spectrum(src_file)   # Load source file of the background
    # Switch off thermal lines in the thermal plasma. Will be added later.
    xspec.Xset.addModelString("APECNOLINES", "yes") 
    ip_model = xspec.Model("tbabs*(apec+gaussian+gaussian+gaussian)")
    unabs_flux = unabs_lx_val/(7.65757E+45)    # Flux at GC distance
    ip_model.setPars({1: nh_val, 2: temp_val, 6: 6.4, 8: 1.0E-4, 9: 6.7,
                      11: 1.0E-4, 12: 7.0, 14: 1.0E-4})  # setting parameters
    # No way to directly specify equivalent width in the model, therefore
    # calculate the equivalenth width given the model and then scale the norm
    # of the model
    xspec.AllModels.eqwidth(3, rangeFrac=0.0)
    test_ew_64 = spectrum.eqwidth[0]
    xspec.AllModels.eqwidth(4, rangeFrac=0.0)
    test_ew_67 = spectrum.eqwidth[0]
    xspec.AllModels.eqwidth(5, rangeFrac=0.0)
    test_ew_70 = spectrum.eqwidth[0]
    norm_64 = ew_64/test_ew_64*1.0E-4
    print(norm_64)
    norm_67 = ew_67/test_ew_67*1.0E-4
    print(norm_67)
    norm_70 = ew_70/test_ew_70*1.0E-4
    print(norm_70)
    # ip_model.setPars({8:0.0, 11:norm_67, 12:0.0})
    # xspec.AllModels.eqwidth(4, rangeFrac=0.0)
    # calc_ew_67 = spectrum.eqwidth[0]
    xspec.AllModels.clear()
    # Reset the parameters again with the calculated norms
    ip_model = xspec.Model("tbabs*cflux*(apec+gaussian+gaussian+gaussian)")
    ip_model.setPars({1: nh_val, 2: 2.0, 3: 10.0, 4: np.log10(unabs_flux),
                      5: temp_val, 9: 6.4, 11: norm_64, 12: 6.7, 14: norm_67,
                      15: 7.0, 17: norm_70})
    xspec.AllData.fakeit(1, ip_settings)
    xspec.AllData.clear()
    xspec.AllModels.clear()
    return norm_64, norm_67, norm_70


def sim_cvs_from_mondal(src_file, sim_cv_name, nh_val, gamma_val,
                        unabs_flux_val, norm_64, norm_67, norm_69):
    """Simulate CVs based on Mondal 2017."""
    ip_settings = xspec.FakeitSettings(fileName=sim_cv_name)
    spectrum = xspec.Spectrum(src_file)
    ip_model = xspec.Model("tbabs*(pegpwrlw+gaussian+gaussian+gaussian)")
    ip_model.setPars({
        1:nh_val/1.0E+22, 2:gamma_val, 3:2.0, 4:10.0, 5:unabs_flux_val,
        6:6.4, 7:0.0, 8:0.0, 9:6.7, 10:0.0, 11:norm_67, 12:6.9, 13:0.0,
        14:0.0})
    xspec.AllModels.eqwidth(4, rangeFrac=0.0)
    ew_67 = spectrum.eqwidth[0]
    print(ew_67)
    ip_model.setPars({
        1:nh_val/1.0E+22, 2:gamma_val, 3:2.0, 4:10.0, 5:unabs_flux_val,
        6:6.4, 7:0.0, 8:norm_64, 9:6.7, 10:0.0, 11:norm_67, 12:6.9, 13:0.0,
        14:norm_69})
    ip_model.show()
    xspec.AllData.fakeit(1, ip_settings)

    xspec.AllData.clear()
    xspec.AllModels.clear()
    return ew_67


def sim_cvs_from_mondal_ews(src_file, sim_cv_name, nh_val, gamma_val,
                            unabs_flux_val, ew_67, i64_i67_ratio,
                            i69_i67_ratio):
    """Using Mondal EWs and line intentisty ratios rather than intensities."""
    ip_settings = xspec.FakeitSettings(fileName=sim_cv_name)
    spectrum = xspec.Spectrum(src_file)
    ip_model = xspec.Model("tbabs*(pegpwrlw+gaussian)")
    ip_model.setPars({1: nh_val, 2: gamma_val, 6:6.7, 7: 0.0, 8:1.0E-7})
    xspec.AllModels.eqwidth(3, rangeFrac=0.0)
    test_ew_67 = spectrum.eqwidth[0]
    norm_67 = ew_67/test_ew_67*1.0E-7
    norm_64 = i64_i67_ratio*norm_67
    norm_69 = i69_i67_ratio*norm_67
    print(norm_64, norm_67, norm_69)
    xspec.AllModels.clear()

    ip_model = xspec.Model("tbabs*cflux*(pegpwrlw+gaussian+gaussian+gaussian)")
    ip_model.setPars({1:nh_val, 2:2.0, 3:10.0, 4:np.log10(unabs_flux_val),
                      5:gamma_val, 9:6.4, 10:0.0, 11:norm_64, 12:6.7, 13:0.0,
                      14:norm_67, 15:6.9, 16:0.0, 17:norm_69})
    ip_model.show()
    xspec.AllData.fakeit(1, ip_settings)

    xspec.AllData.clear()
    xspec.AllModels.clear()
    return norm_67


def sim_cv_from_craig(src_file, sim_msp_name, nh_val, gamma_val, unabs_lx_val,
                      ew_fe):
    """This was from one of Craig's results, where is used one Gaussian."""
    ip_settings = xspec.FakeitSettings(fileName=sim_msp_name)
    spectrum = xspec.Spectrum(src_file)
    ip_model = xspec.Model("tbabs*(pegpwrlw+gaussian)")
    unabs_flux = unabs_lx_val/(7.65757E+45)
    ip_model.setPars({1: nh_val, 2: gamma_val, 5: unabs_flux/1.0E-12, 6: 6.54,
                      7: 0.2, 8: 1.0E-4})
    xspec.AllModels.eqwidth(3, rangeFrac=0.0)
    test_ew = spectrum.eqwidth[0]
    norm = ew_fe/test_ew*1.0E-4
    xspec.AllModels.clear()
    ip_model = xspec.Model("tbabs*(pegpwrlw+gaussian)")
    ip_model.setPars({1: nh_val, 2: gamma_val, 5: unabs_flux/1.0E-12, 6: 6.54,
                      7: 0.2, 8: norm})
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
    pn_src_files = glob2.glob(src_folder + '*_PN_*grp1_src.ds')
    mos_src_files = glob2.glob(src_folder + '*_MOS_*grp1_src.ds')
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
                                  '_MOS_combined_grp1_src.ds')[0]
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


def msp_sims_chandra_src(num_msps, nh_vals, gamma_vals, unabs_lx_vals,
                         src_folder, sim_msp_folder='./', file_prefix='msp_'):
    """Simulate Chandra MSPs.
    
    Give the absolute path for sim_msp_folder.
    """
    src_files = glob2.glob(src_folder + '/*/*_combined_src.pi')
    src_args = np.random.choice(np.arange(len(src_files)), size=num_msps)
    for i in range(num_msps):
        sim_msp_from_src(
            src_files[src_args[i]],
            sim_msp_folder + file_prefix + str(i) + '.fak',
            nh_vals[i], gamma_vals[i], unabs_lx_vals[i])
        
        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')


def cvs_sims_chandra_src(num_msps, nh_vals, temp_vals, unabs_lx_vals,
                         ew_64_vals, ew_67_vals, ew_70_vals, src_folder,
                         sim_cv_folder='./', file_prefix='ip_'):
    """Simulate Chandra MSPs.
    
    Give the absolute path for sim_msp_folder.
    """
    src_files = glob2.glob(src_folder + '/*/*_combined_src.rmf')
    src_args = np.random.choice(np.arange(len(src_files)), size=num_msps)
    curr_dir = os.getcwd()
    norm_vals = np.zeros((num_msps, 3), dtype=float)
    for i in range(num_msps):
        os.chdir(os.path.dirname(src_files[src_args[i]])) 
        srcfilename = os.path.basename(src_files[src_args[i]])[:-3] + 'pi'
        norm_vals[i] = sim_cv_from_src(
            srcfilename,
            sim_cv_folder + file_prefix + str(i) + '.fak',
            nh_vals[i], temp_vals[i], unabs_lx_vals[i], ew_64_vals[i],
            ew_67_vals[i], ew_70_vals[i])
        os.chdir(curr_dir)
        
        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')
    return norm_vals


def cvs_sims_chandra_src_pl(num_msps, nh_vals, gamma_vals, unabs_flux_vals,
                            ew_67_vals, ratios_64_67, ratios_69_67,
                            src_folder, sim_cv_folder='./', file_prefix='cv_'):
    """Simulate Chandra MSPs.
    
    Give the absolute path for sim_msp_folder.
    """
    src_files = glob2.glob(src_folder + '/*/*_combined_src.rmf')
    src_args = np.random.choice(np.arange(len(src_files)), size=num_msps)
    curr_dir = os.getcwd()
    norm_67_vals = np.zeros(num_msps, dtype=float)
    for i in range(num_msps):
        os.chdir(os.path.dirname(src_files[src_args[i]])) 
        srcfilename = os.path.basename(src_files[src_args[i]])[:-3] + 'pi'
        norm_67_vals[i] = sim_cvs_from_mondal_ews(
            srcfilename,
            sim_cv_folder + file_prefix + str(i) + '.fak',
            nh_vals[i], gamma_vals[i], unabs_flux_vals[i], ew_67_vals[i],
            ratios_64_67[i], ratios_69_67[i])
        os.chdir(curr_dir)
        
        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')
    return norm_67_vals


def msp_sims_from_src2(num_msps, nh_vals, gamma_vals, unabs_lx_vals,
                       src_folder, sim_msp_folder='./', file_prefix='msp_'):
    """Simulate equal number of PN and MOS MSPs from source files."""
    (common_files_pn, common_files_mos, only_pn_files,
     only_mos_files) = get_xmm_src_files(src_folder)
    pn_files = common_files_pn + only_pn_files
    mos_files = common_files_mos + only_mos_files
    pn_src_args = np.random.choice(np.arange(len(pn_files)), size=num_msps)
    mos_src_args = np.random.choice(np.arange(len(mos_files)), size=num_msps)
    for i in range(num_msps):
        sim_msp_from_src(
                pn_files[pn_src_args[i]],
                sim_msp_folder + file_prefix + str(i) + '_PN.fak',
                nh_vals[i], gamma_vals[i], unabs_lx_vals[i])
        sim_msp_from_src(
                mos_files[mos_src_args[i]],
                sim_msp_folder + file_prefix + str(i) + '_MOS.fak',
                nh_vals[i], gamma_vals[i], unabs_lx_vals[i])

        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')
    

def msp_sims_from_src(num_msps, nh_vals, gamma_vals, unabs_lx_vals, src_folder,
                      sim_msp_folder='./', file_prefix='msp_'):
    """Simulate PN and MOS MSPs from source files based on source detection."""
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


def cvs_sims_from_src2(num_msps, nh_vals, temp_vals, unabs_lx_vals, ew_64_vals,
                       ew_67_vals, ew_70_vals, src_folder, sim_cv_folder='./',
                       file_prefix='cv_'):
    """Simulate equal number of PN and MOS MSPs from source files."""
    (common_files_pn, common_files_mos, only_pn_files,
     only_mos_files) = get_xmm_src_files(src_folder)
    pn_files = common_files_pn + only_pn_files
    mos_files = common_files_mos + only_mos_files
    norm_vals = np.zeros((num_msps, 3), dtype=float)
    pn_src_args = np.random.choice(np.arange(len(pn_files)), size=num_msps)
    mos_src_args = np.random.choice(np.arange(len(mos_files)), size=num_msps)
    for i in range(num_msps):
        norm_vals[i] = sim_cv_from_src(
                pn_files[pn_src_args[i]],
                sim_cv_folder + file_prefix + str(i) + '_PN.fak',
                nh_vals[i], temp_vals[i], unabs_lx_vals[i], ew_64_vals[i],
                ew_67_vals[i], ew_70_vals[i])
        sim_cv_from_src(
                mos_files[mos_src_args[i]],
                sim_cv_folder + file_prefix + str(i) + '_MOS.fak',
                nh_vals[i], temp_vals[i], unabs_lx_vals[i], ew_64_vals[i],
                ew_67_vals[i], ew_70_vals[i])

        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')
    return norm_vals


def cvs_sims_from_src_pl(num_cvs, nh_vals, gamma_vals, unabs_flux_vals,
                         ew_67_vals, ratios_64_67, ratios_69_67,
                         src_folder, sim_cv_folder='./', file_prefix='cv_'):
    """Simulate CVs for a power law model, specifically Mondal paper"""
    (common_files_pn, common_files_mos, only_pn_files,
     only_mos_files) = get_xmm_src_files(src_folder)
    pn_files = common_files_pn + only_pn_files
    mos_files = common_files_mos + only_mos_files
    pn_src_args = np.random.choice(np.arange(len(pn_files)), size=num_cvs)
    mos_src_args = np.random.choice(np.arange(len(mos_files)), size=num_cvs)
    norm_67_vals = np.zeros(num_cvs, dtype=float)
    for i in range(num_cvs):
        norm_67_vals[i] = sim_cvs_from_mondal_ews(
                pn_files[pn_src_args[i]],
                sim_cv_folder + file_prefix + str(i) + '_PN.fak',
                nh_vals[i], gamma_vals[i], unabs_flux_vals[i], ew_67_vals[i],
                ratios_64_67[i], ratios_69_67[i])
        sim_cvs_from_mondal_ews(
                mos_files[mos_src_args[i]],
                sim_cv_folder + file_prefix + str(i) + '_MOS.fak',
                nh_vals[i], gamma_vals[i], unabs_flux_vals[i], ew_67_vals[i],
                ratios_64_67[i], ratios_69_67[i])

        if i % 1000 == 0:
            print('Finished ' + str(i) + ' simulations')

    return norm_67_vals


def get_cv_params_mondal(num_cvs):
    """Get CV parameter values"""
    nh_vals = 10**np.random.uniform(22.7, 23.7, num_cvs)
    gamma_giv = [0.43, 0.72, 0.22, 0.0, 0.62, 0.38, 0.62, 0.96, 1.37, 0.56,
                 0.26, 0.98, 0.11, 0.28, -0.70, 1.16, 0.11, 0.96, -0.73, 0.19,
                 1.24, -0.17, 0.23, 0.35, 0.91, 0.37, 0.24, 0.68, 0.74, 0.0,
                 0.67, 0.63, 0.67, -0.43, 0.61, -0.67, -0.41, -0.31, 0.31]
    i67_giv = [2.51, 0.79, 3.50, 17.0, 5.92, 4.22, 2.24, 2.48, 3.04, 1.16,
               1.52, 3.35, 2.66, 1.10, 6.84, 0.64, 1.93, 1.62, 1.98, 1.84,
               1.59, 14.0, 1.20, 1.17, 1.83, 4.27, 6.89, 2.09, 3.17, 1.14,
               2.56, 1.64, 1.95, 4.01, 2.11, 1.42, 0.29, 2.16, 2.71]
    i69_giv = [0.0, 0.0, 2.69, 10.9, 6.57, 3.51, 1.15, 2.23, 2.48, 0.0, 0.93,
               2.24, 0.0, 0.0, 5.34, 0.0, 1.89, 1.19, 1.04, 1.89, 1.49, 5.85,
               0.90, 0.0, 1.56, 2.96, 0.68, 1.80, 0.0, 0.0, 1.36, 1.43, 1.40,
               4.01, 1.56, 0.90, 0.0, 1.76, 1.59]
    flux_giv = [15.3, 1.92, 4.30, 38.0, 34.0, 19.6, 3.36, 10.4, 6.27, 3.85,
                5.25, 1.12, 1.26, 1.74, 8.69, 2.77, 5.96, 2.23, 1.69, 8.92,
                1.51, 60.1, 1.34, 1.73, 2.46, 9.21, 6.45, 2.09, 12.9, 3.08,
                3.91, 3.35, 4.83, 11.1, 5.27, 2.40, 4.63, 4.57, 6.46]
    ratio_64_67_giv = np.random.choice(
        [0.68, 1.18, 0.13, 1.75, 1.26, 1.5, 1.21, 1.17, 1.93, 1.83, 0.86, 0.71,
         1.29, 1.24, 1.98, 0.76, 0.84], num_cvs)
    random_index = np.random.choice(np.arange(39), num_cvs)
    gamma_vals = np.array(gamma_giv)[random_index]
    flux_vals = np.array(flux_giv)[random_index]*0.1
    norm_67vals = np.array(i67_giv)[random_index]*1.0E-6
    norm_69vals = np.array(i69_giv)[random_index]*1.0E-6
    norm_64vals = norm_67vals*ratio_64_67_giv
    return (nh_vals, gamma_vals, flux_vals, norm_64vals, norm_67vals,
            norm_69vals)
    

def cv_params_mondal2(num_cvs):
    """Get CV parameters."""
    nh_vals = 10**np.random.uniform(22.7, 23.7, num_cvs)
    lx_vals = 10**np.random.uniform(31.0, 34.0, num_cvs)
    flux_vals = lx_vals/(7.657569170326442e+45)
    gamma_giv = [0.43, 0.72, 0.22, 0.0, 0.62, 0.38, 0.62, 0.96, 1.37, 0.56,
                 0.26, 0.98, 0.11, 0.28, -0.70, 1.16, 0.11, 0.96, -0.73, 0.19,
                 1.24, -0.17, 0.23, 0.35, 0.91, 0.37, 0.24, 0.68, 0.74, 0.0,
                 0.67, 0.63, 0.67, -0.43, 0.61, -0.67, -0.41, -0.31, 0.31]
    ew_vals_giv = [0.13, 0.33, 0.62, 0.35, 0.14, 0.17, 0.55, 0.19, 0.46, 0.23,
                   0.19, 0.25, 0.16, 0.54, 0.89, 0.21, 0.24, 0.67, 1.01, 0.15,
                   1.13, 0.18, 0.53, 0.50, 0.55, 0.35, 0.74, 0.82, 0.19, 0.27,
                   0.57, 0.35, 0.31, 0.3, 0.32, 0.41, 0.49, 0.38, 0.31]
    ratios_69_67_giv = [0.00, 0.00, 0.77, 0.64, 1.11, 0.83, 0.51, 0.90, 0.82,
                        0.00, 0.61, 0.67, 0.00, 0.00, 0.78, 0.00, 0.98, 0.73,
                        0.53, 1.03, 0.94, 0.42, 0.75, 0.00, 0.85, 0.69, 0.10,
                        0.86, 0.00, 0.00, 0.53, 0.87, 0.72, 0.73, 0.74, 0.63,
                        0.00, 0.81, 0.59]
    ratios_64_67 = np.random.choice(
        [0.68, 1.18, 0.13, 1.75, 1.26, 1.5, 1.21, 1.17, 1.93, 1.83, 0.86, 0.71,
         1.29, 1.24, 1.98, 0.76, 0.84], num_cvs)
    random_index = np.random.choice(np.arange(39), num_cvs)
    gamma_vals = np.array(gamma_giv)[random_index]
    ew_67_vals = np.array(ew_vals_giv)[random_index]
    ratios_69_67 = np.array(ratios_69_67_giv)[random_index]
    return (nh_vals, gamma_vals, flux_vals, ew_67_vals, ratios_64_67,
            ratios_69_67)
    


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
    """"Get parameter values for IP simulations.
    
    nh_vals = Currently using a log uniform distribution
    lx_vals - Currently using a log uniform distribution
    temp_vals, ew_vals - Using normal distribution. Might need to change

    """
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
        temp_vals = np.random.normal(34.0, 14.61, num_cvs)
        ew_64_vals = np.random.normal(115.0, 36.22, num_cvs)*2
        ew_67_vals = np.random.normal(107, 65.39, num_cvs)*2
        ew_70_vals = np.random.normal(80, 27.91, num_cvs)*2
    elif cv_type == 'SS':
        temp_vals = np.random.normal(27.2, 20.8, num_cvs)
        ew_64_vals = np.random.normal(280, 90, num_cvs)
        ew_67_vals = np.random.normal(241, 78.3, num_cvs)
        ew_70_vals = np.random.normal(91, 20.1, num_cvs)
    else:
        print('CV types can only be IP or SS')
    return nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals, ew_70_vals

def ip_param_vals(num_cvs, nh_abs_type='high'):
    """Get IP parameter values.

    Same as previous, but picking the temp and EW values randomly from the list
    rather than using a normal distribution.
    """
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

    temp_vals = np.random.choice(
        [19.7, 42.6, 9.41, 19.1, 30.5, 63.6, 15.8, 43.5, 32.6, 64.0, 40.5,
         26.9, 26.6, 22.8, 47.3, 39.6, 31.6], num_cvs)
    ew_64_vals = np.random.choice(
        [158, 102, 32, 133, 128, 88, 139, 156, 128, 172, 120, 88, 97, 140, 131,
         70], num_cvs)/1000
    ew_67_vals = np.random.choice(
        [174, 81, 325, 73, 91, 59, 101, 116, 68, 79, 131, 121, 71, 102, 60,
         94], num_cvs)/1000
    ew_70_vals = np.random.choice(
        [100, 54, 110, 58, 32, 62, 134, 120, 62, 91, 104, 94, 70, 93, 69, 57],
        num_cvs)/1000
    return nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals, ew_70_vals


def ip_param_vals_craig(num_cvs):
    """Param values of ASCA IPs."""
    nh_vals = 10**np.random.uniform(22.7, 23.7, num_cvs)
    lx_vals = 10**np.random.uniform(31.0, 34.0, num_cvs)
    gamma_vals = np.random.choice(
        [2.53, 1.32, 0.9, 1.32, 1.98, 0.66, 1.29, 0.59, 1.08, 1.0, 1.11, 1.83,
         0.81, 1.07, 0.86, 1.2, 0.95, 1.49, 1.12, 1.96], num_cvs)
    ew_vals = np.random.choice(
        [769, 450, 4.12, 264, 772, 488, 206, 596, 311, 388, 403, 456, 298, 282,
         247, 389, 411, 691, 206, 698], num_cvs)/1000
    return nh_vals, gamma_vals, lx_vals, ew_vals


def main_cvs_xu(num_cvs=10000, src_folder=None, sim_cv_folder=None,
                telescope='XMM', sim_cv_folder2=None):
    """Generate CVs according to Xu model."""
    (nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals,
     ew_70_vals) = ip_param_vals(num_cvs)
    if src_folder is None:
        src_folder = './'
    if sim_cv_folder is None:
        sim_cv_folder = './'

    if telescope == 'XMM':
        norm_vals = cvs_sims_from_src2(
            num_cvs, nh_vals/1.0E+22, temp_vals, lx_vals, ew_64_vals,
            ew_67_vals, ew_70_vals, src_folder, sim_cv_folder)
        cv_param_vals = np.column_stack([
            nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals,
            ew_70_vals, norm_vals[:, 0], norm_vals[:, 1], norm_vals[:, 2]])
        np.savetxt(sim_cv_folder + 'param_cvs.txt', cv_param_vals)
        if sim_cv_folder2 is not None:
            norm_vals_2 = cvs_sims_from_src2(
                num_cvs, nh_vals/1.0E+22, temp_vals, lx_vals, ew_64_vals*0.5,
                ew_67_vals*0.5, ew_70_vals*0.5, src_folder, sim_cv_folder2)
            cv_param_vals = np.column_stack([
                nh_vals, temp_vals, lx_vals, ew_64_vals*0.5, ew_67_vals*0.5,
                ew_70_vals*0.5, norm_vals_2[:, 0], norm_vals_2[:, 1],
                norm_vals_2[:, 2]])
            np.savetxt(sim_cv_folder2 + 'param_cvs.txt', cv_param_vals)
    else:
        norm_vals = cvs_sims_chandra_src(
            num_cvs, nh_vals/1.0E+22, temp_vals, lx_vals, ew_64_vals, ew_67_vals,
            ew_70_vals, src_folder, sim_cv_folder)
        cv_param_vals = np.column_stack([
            nh_vals, temp_vals, lx_vals, ew_64_vals, ew_67_vals, ew_70_vals,
            norm_vals[:, 0], norm_vals[:, 1], norm_vals[:, 2]])
        np.savetxt(sim_cv_folder + 'param_cvs.txt', cv_param_vals)
        if sim_cv_folder2 is not None:
            norm_vals_2 = cvs_sims_chandra_src(
                num_cvs, nh_vals/1.0E+22, temp_vals, lx_vals, ew_64_vals*0.5,
                ew_67_vals*0.5, ew_70_vals*0.5, src_folder, sim_cv_folder2)
            cv_param_vals = np.column_stack([
                nh_vals, temp_vals, lx_vals, ew_64_vals*0.5, ew_67_vals*0.5,
                ew_70_vals*0.5, norm_vals_2[:, 0], norm_vals_2[:, 1],
                norm_vals_2[:, 2]])
            np.savetxt(sim_cv_folder2 + 'param_cvs.txt', cv_param_vals)
    
    return cv_param_vals
            


def main_cvs(num_cvs=10000, src_folder=None,
             sim_cv_folder='./'):
    """Generate CVs"""
    (nh_vals, gamma_vals, flux_vals, ew_67vals, ratios_6467,
     ratios_6967) = cv_params_mondal2(num_cvs)
    
    if src_folder is None:
        src_folder = ('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                      'data/xmm_combined_goodobs2')
    norm_67vals = cvs_sims_from_src_pl(
        num_cvs, nh_vals/1.0E+22, gamma_vals, flux_vals, ew_67vals,
        ratios_6467, ratios_6967, src_folder, sim_cv_folder)
    # norm_67vals = cvs_sims_chandra_src_pl(
    #    num_cvs, nh_vals/1.0E+22, gamma_vals, flux_vals, ew_67vals,
    #    ratios_6467, ratios_6967, src_folder, sim_cv_folder)
    cv_param_vals = np.column_stack([
        nh_vals, gamma_vals, flux_vals, ew_67vals, ratios_6467, ratios_6967,
        norm_67vals])
    
    np.savetxt(sim_cv_folder + 'param_cvs.txt', cv_param_vals)
    return cv_param_vals


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
    return nh_vals, gamma_vals, lx_vals