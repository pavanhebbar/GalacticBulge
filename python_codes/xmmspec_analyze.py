"""Program to analyze the spectra of 4XMM DR11 Galactic Bulge sources."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from astropy.io import fits


class Autoencoder(tf.keras.models.Model):
    """Defining the autoencoder."""
    def __init__(self, in_dim=526, latent_dim=10, encoder=None, decoder=None):
        """Initialize function"""
        super(Autoencoder, self).__init__()
        self.input_dim = in_dim
        self.latent_dim = latent_dim
        if encoder is None:
            self.encoder = tf.keras.Sequential([
                layers.Dense(latent_dim, activation='relu')])
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = tf.keras.Sequential([
                layers.Dense(in_dim, activation=layers.LeakyReLU(alpha=0.01))])
        else:
            self.decoder = decoder
 
    def _call(self, input_x):
        """Call function."""
        encoded = self.encoder(input_x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, input_x):
        """Encode input data."""
        return self.encoder(input_x)

    def decode(self, encoded_input):
        """Decode encoded data."""
        return self.decoder(encoded_input)


testing = 2


def cstat_loss(source, model, background=None, bg_model=None, backscal=):
    """Poission loss.
    
    Assumes same exposure times for source and the background
    """
    if background is None:
        zero_counts = np.where(source == 0)
        model[np.where(model==0)] = 1.0E-10
        cstat_arr = model - source + source*np.log(source/model)
        cstat_arr[zero_counts] = model
        return 2.0*np.sum(cstat_arr)
    
    if bg_model is None:
        zero_src_counts = np.where(source == 0)
        zero_bg_counts_1 = np.where(
            np.logical_and(background==0, model < 0.5*source))
        zero_bg_counts_2 = np.where(
            np.logical_and(background==0, model >= 0.5*source))

        d_arr = ((2*model - source - background)**2 + 8*model*background)**0.5
        bg_model = (source + background - 2*model + d_arr)/4.0
    
    wstat_arr = 2*(model + 2*bg_model - source*(np.log(model+bg_model) + 1.0
                                                - np.log(source))
                  - background*(np.log(bg_model) + 1.0 + np.log(background)))
    


def get_energybins(resp_file):
    """Extract the energy bins."""
    channel_ebounds = fits.getdata(resp_file, 2)
    e_min = channel_ebounds['e_min']
    e_max = channel_ebounds['e_max']
    e_bins = np.append(e_min, e_max[-1])
    ebin_centres = 0.5*(e_min + e_max)
    return e_bins, ebin_centres


class Spectra:
    """Spectra class."""
    def __init__(self, src_spectra, bg_spectra=None, rmf=None, arf=None,
                 ebins=None, spec_prop=None, en_range=None, backscal=1.0):
        """Initialize class."""
        if rmf is None and ebins is None :
            raise ValueError('Need to give rmf to calculate energy bins or' +
                             'specify enery bins')
        elif ebins is None:
            ebins, e_centres = get_energybins(rmf)
        else: 
            e_centres = 0.5*(ebins[:-1] + ebins[1:])
        self.all_ebins = ebins
        self.full_spec = src_spectra
        
        self._ebins = ebins[start_index:end_index+1]
        self._ebin_centres = e_centres[start_index:end_index]
        self._src_spec = src_spectra[start_index:end_index]
        self._src_counts = np.sum(self._src_spec)*1.0
        if bg_spectra is None:
            self._bg_spec = np.zeros_like(self._src_spec)*backscal
            self._bg_counts = 0.0
        else:
            self._bg_spec = bg_spectra[start_index:end_index]*backscal
            self._bg_counts = np.sum(self._bg_spec)*1.0
        self._net_spec =  self._src_spec - self._bg_spec
        self._net_counts = self._src_counts - self._bg_counts
        self._bg_to_net_ratio = self._bg_counts/self._net_counts
        self.resp = rmf
        self.arf = arf
        if ebins is None and rmf is None:
            raise ValueError("No way to generate energy bins")
        elif ebins is not None:
            pass
        else:
            pass

    def _get_startendindex(self, en_range=None):
        """Get start and end index for given en_range."""
        if en_range is None:
            en_range = (0.3, 8.0)
        start_index = np.where(self.all_ebins >= en_range[0])[0][0]
        end_index = np.where(self.all_ebins <= en_range[1])[0][-1]
        return start_index, end_index

    def get_netspec(self):
        """Get net spectra."""
        return self._net_spec
    
    def get_counts(self):
        """Return source, scaled bg counts and net counts"""
        return self._src_counts, self._bg_counts, self._net_counts
