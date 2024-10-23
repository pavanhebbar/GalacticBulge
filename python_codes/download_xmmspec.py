"""Program to download XMM data."""


import sys
import subprocess
import numpy as np


def load_table(table_text_file, delimiter=',', skiprows=5, obselete_columns=0):
    """Load table."""
    obs_table = np.loadtxt(table_text_file, dtype=str, delimiter=delimiter,
                           skiprows=skiprows)
    return obs_table[:, obselete_columns:]


def get_file_url_list(directory_url, file_prefix, file_suffix, pattern_lists,
                      exclude_pattern=None):
    """Get the list of file URLs matching the pattern."""

    directory_contents = subprocess.run([
        'wget', '-nv', '-O-', directory_url], capture_output=True, check=False)
    file_lists = []
    for pattern_list in pattern_lists:
        file_pattern = '"' + file_prefix
        for pattern in pattern_list:
            file_pattern = file_pattern + '[^ ]*' + pattern
        file_pattern = file_pattern + '[^ ]*' + file_suffix + '"'
        print(file_pattern)
        if exclude_pattern is None:
            file_list = subprocess.run(
                ['grep', '-o', file_pattern],
                input=directory_contents.stdout, capture_output=True,
                check=False).stdout.decode().split('"\n"')
        else:
            file_list_full = subprocess.run(
                ['grep', '-o', file_pattern],
                input=directory_contents.stdout, capture_output=True,
                check=False)
            file_list = subprocess.run(
                ['grep', '-v', exclude_pattern],
                input=file_list_full.stdout, capture_output=True,
                check=False).stdout.decode().split('"\n"')
        file_list[0] = file_list[0][1:]
        file_list[-1] = file_list[-1][:-2]
        if directory_url[-1] == '/':
            directory_url = directory_url[:-1]
        for j, file in enumerate(file_list):
            if file != '':
                file_list[j] = directory_url + '/' + file
        file_lists.append(file_list)
    return file_lists


def download_spec(det_id_list, source_id_list, source_num_list,
                  download_dir='./', obs_id_list=None):
    """Download spectra."""
    source_id = '0'
    for i, detid in enumerate(det_id_list):
        if obs_id_list is None:
            obs_id = detid[1:11]
        else:
            obs_id = obs_id_list[i]
        source_num = int(source_num_list[i])
        source_num_str_hex = format(source_num, '04X')
        if source_id_list[i] != source_id:
            source_id = source_id_list[i]
            subprocess.run(['mkdir', download_dir+source_id], check=False)
            subprocess.run(['mkdir', download_dir+source_id + '/EPIC_PN_spec'],
                           check=False)
            subprocess.run(['mkdir', download_dir+source_id +
                            '/EPIC_MOS1_spec'],
                           check=False)
            subprocess.run(['mkdir', download_dir+source_id +
                            '/EPIC_MOS2_spec'],
                           check=False)

        # Get URL of spectra.
        spec_url_list = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/' +
                           obs_id + '/PPS/'),
            file_prefix='P' + obs_id, file_suffix='.FTZ',
            pattern_lists=[['M1', 'SRSPEC'+source_num_str_hex],
                           ['M2', 'SRSPEC'+source_num_str_hex],
                           ['PN', 'SRSPEC'+source_num_str_hex]])
        spec_image_urls = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/' +
                           obs_id + '/PPS/'),
            file_prefix='P' + obs_id, file_suffix='.PDF',
            pattern_lists=[['M1', 'SPCPLT'+source_num_str_hex],
                           ['M2', 'SPCPLT'+source_num_str_hex],
                           ['PN', 'SPCPLT'+source_num_str_hex]])
        spec_regions_urls = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/' +
                           obs_id + '/PPS/'),
            file_prefix='P' + obs_id, file_suffix='.PNG',
            pattern_lists=[['M1', 'SRSPEC'+source_num_str_hex],
                           ['M2', 'SRSPEC'+source_num_str_hex],
                           ['PN', 'SRSPEC'+source_num_str_hex]])
        bgspec_url_list = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/' +
                           obs_id + '/PPS/'),
            file_prefix='P' + obs_id, file_suffix='.FTZ',
            pattern_lists=[['M1', 'BGSPEC'+source_num_str_hex],
                           ['M2', 'BGSPEC'+source_num_str_hex],
                           ['PN', 'BGSPEC'+source_num_str_hex]])
        arf_url_list = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/' +
                           obs_id + '/PPS/'),
            file_prefix='P' + obs_id, file_suffix='.FTZ',
            pattern_lists=[['M1', 'SRCARF'+source_num_str_hex],
                           ['M2', 'SRCARF'+source_num_str_hex],
                           ['PN', 'SRCARF'+source_num_str_hex]])

        # Download spectra
        det_names = ['/EPIC_MOS1_spec', '/EPIC_MOS2_spec', '/EPIC_PN_spec']
        for j, spec_urls in enumerate(spec_url_list):
            for k, spec_url in enumerate(spec_urls):
                print(spec_url)
                if spec_url != '':
                    subprocess.run(['wget', '-P',
                                    download_dir+source_id+det_names[j], '-c',
                                    spec_url], check=False)
                    subprocess.run(['wget', '-P',
                                    download_dir+source_id+det_names[j], '-c',
                                    bgspec_url_list[j][k]], check=False)
                    subprocess.run(['wget', '-P',
                                    download_dir+source_id+det_names[j], '-c',
                                    arf_url_list[j][k]], check=False)
                if spec_image_urls[j][k] != '':
                    subprocess.run(['wget', '-P',
                                    download_dir+source_id+det_names[j], '-c',
                                    spec_image_urls[j][k]], check=False)
                if spec_regions_urls[j][k] != '':
                    subprocess.run(['wget', '-P',
                                    download_dir+source_id+det_names[j], '-c',
                                    spec_regions_urls[j][k]], check=False)


def download_resp(download_dir):
    """Download the XMM_newton response files."""
    subprocess.run(['mkdir', download_dir+'PN'], check=False)
    subprocess.run(['mkdir', download_dir+'MOS'], check=False)
    det_types = ['PN', 'MOS', 'MOS']
    det_prefixes = ['epn', 'm1', 'm2']
    resp_url_lists = []
    for i, prefix in enumerate(det_prefixes):
        url_list = get_file_url_list(
            directory_url=('https://heasarc.gsfc.nasa.gov/FTP/xmm/data/' +
                           'responses/' + det_types[i]),
            file_prefix=prefix+'_', file_suffix='',
            pattern_lists=['.rmf'])
        resp_url_lists.append(url_list)

    print(len(resp_url_lists))
    for i, url_list in enumerate(resp_url_lists):
        print(len(url_list[0]))
        for url in url_list[0]:
            print(url)
            subprocess.run(['wget', '-P', download_dir+det_types[i], '-c',
                            url], check=False)


def main(table_loc, download_loc, rows_to_skip=4, delimiter='\t'):
    """Default run function."""
    det_table = load_table(table_loc, delimiter=delimiter,
                           skiprows=rows_to_skip)
    detid_list = det_table[:, 0].astype(int).astype(str).tolist()
    srcid_list = det_table[:, 1].astype(int).astype(str).tolist()
    srcnum_list = det_table[:, 3].astype(int).astype(str).tolist()
    obsid_list = det_table[:, 4].astype(int).astype(str).tolist()
    obsid_list = ['0' + obs_id for obs_id in obsid_list]
    print(obsid_list)
    download_spec(detid_list, srcid_list, srcnum_list, download_loc,
                  obsid_list)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
