from __future__ import division

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl


def get_freq_statistics(freq_selected, psds):

        """
        Explore statistics for distribution of all PSDs at one frequency bin

        Given a set of PSDs computed from several frequency chunks of the 
        same data set, the spread of strain / sqrt(Hz) values are compared 
        for all PSDs at a selected frequency. Mean, variance, skewness, and
	kurtosis are returned.
        
        freq_selected --  a frequency in Hz at which the PSDs will be compared

        psds -- a list of tuples, with one tuple representing the frequency
        and strain / sqrt(Hz) values for one PSD

        mean -- average of the strain / sqrt(Hz) a.k.a. psd_h values at 
        the selected frequency

        var -- variance of the aforementioned values

	skew -- third order statistics

	kurt -- fourth order statistics
        """

        # Store psd_h for selected frequncy in each PSD
        psd_h = []

        # Find corresponding PSD value for selected frequency
        for psd in psds:
                for index, freq in enumerate(psd[0]):
                        if freq == freq_selected:
                                psd_h.append(psd[1][index])
                                # !!! - awkward to have list of arrays


        # Calculate moments of power for corresponding psd_h
	n, min_max, mean, var, skew, kurt = stats.describe(psd_h)       
 
	return mean, var, skew, kurt


def plot_histogram(freq_selected, psds):
	
	"""
	Plot a histogram of various PSD values at one selected frequency.

	freq_selected -- selected frequency in Hz

	psds -- a list of tuples, with one tuple representing the frequency
	and strain / sqrt(Hz) values for one PSD
	"""

	# Store psd_h for selected frequency in each PSD
	psd_h = []	

	# Find corresponding PSD value for each selected frequency
	for psd in psds:
		for index, freq in enumerate(psd[0]):
			if freq == freq_selected:
				psd_h.append(psd[1][index])

	# Plot histogram
	plt.figure(5)
	plt.hist(psd_h, 20)
	plt.xlabel('PSD (strain / (Sqrt(Hz))')
	plt.ylabel('Count')
	plt.savefig('histogram.pdf')
	

def plot_timeseries(time_seg, strain_seg):
	
	"""
	Given input data of strain vs. time, a timeseries is plotted
	
	time_seg -- a list of time values

	strain_seg -- a list of strain values corresponding to time values 
	in the same index of time_seg; same length as time_seg

	"""

	plt.figure(1)
	plt.plot(time_seg - time_seg[0], strain_seg)
	plt.xlabel('Time since GPS ' + str(time_seg[0]) + ' (s)')
	plt.ylabel('Strain')
	plt.title('Time Series')
	plt.savefig('timeseries.pdf')


def plot_psd(time_seg, strain_seg, fs, plotting=True):
	
	"""
	Creates a power spectral density plot for a set of strain values

	The PSD is calculated using the matplotlib.mlab.psd() function
	which uses a Hanning window function by default.

	NFFT is set to the sampling frequency. 

	Overlap is half of NFFT.	
	
	time_seg -- a list of time values

	strain_seg -- a list of strain values corresponding to time values 
	in the same index of time_seg; same length as time_seg

	fs -- sampling frequency of data in Hz

	plotting -- Boolean; if true plots are produced
	
	pxx -- PSD values corresponding to strain values

	freqs -- frequencies at which a PSD amplitude is calculated


	"""

	pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=fs, noverlap=fs/2)
	if plotting:
		plt.figure(2)	
		plt.loglog(freqs, np.sqrt(pxx))
		plt.grid('on')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('PSD (strain /  Sqrt(Hz))')
		plt.title('PSD for L1 data starting at GPS ' + str(time_seg[0]))
		plt.ylim([1e-26, 1e-16])
		plt.savefig('original_PSD.pdf')
	return pxx, freqs

########
########
########
# Main code starts from here

########
# Read in data

fname = 'L-L1_LOSC_4_V1-842657792-4096.hdf5'
print "=== Start reading data ==="
print "* File name: ", fname

strain, time, channel_dictionary = rl.loaddata(fname)
print "* Data reading done."

data_len = len(time);
time_spacing = time[1] - time[0]
fs = int(1/time_spacing)

print "* Number of samples:", data_len, " = 2^", np.log2(data_len)
print "* Start GPS Time:", time[0]
print "* End GPS Time:", time[data_len-1]
print "* Sampling Time", time_spacing, " sec = ", fs, " Hz"
print "" # blank line

########
# Select a data segment without errors ("good data")
print "=== DQ good segments ==="
seglist = rl.dq_channel_to_seglist(channel_dictionary['DEFAULT'], fs)
num_seg = len(seglist);
print "* Number of unintruppted 'good' segments", num_seg

i_seg = 0;
print "* Use segment #", i_seg
time_seg = time[seglist[0]]
time_seg_max = max(time_seg)
time_seg_min = min(time_seg)
print "* Length of the segment ", time_seg_max - time_seg_min, "sec =", (time_seg_max - time_seg_min)*fs, "samples"
print "" # blank line

########
# Extract a chunk of data from the selected segment
print "=== Extracting a chunk of data from the selected segment ==="
chunk_length_sec = 16  # sec
chunk_length     = chunk_length_sec*fs # number of samples
print "* Chunk length ", chunk_length_sec, " = ", chunk_length, "saples"

#i_chunk = 2;
for i_chunk in range(0,10):
     print "* Chunk #", i_chunk
     strain_chunk = strain[seglist[i_seg]][chunk_length*i_chunk:chunk_length*(i_chunk+1)]
     time_chunk   =   time[seglist[i_seg]][chunk_length*i_chunk:chunk_length*(i_chunk+1)]

     print "* Extracted chunk length", len(time_chunk)
     print "" # blank line

     # Plot a time series
     # plot_timeseries(time_chunk, strain_chunk)

########
# Calculate a PSD
     print "=== Caluculating the power spectral density for the chunk ==="

     # Choose one from the following windows
     # my_window = mlab.window_none(np.ones(chunk_length)))
     my_window = mlab.window_hanning(np.ones(chunk_length))

     # matplotlib detrend default is 'none'. Leave as it is.
     pxx, freqs = mlab.psd(strain_chunk, Fs=fs, NFFT=chunk_length, window=my_window);

     f_resolution = freqs[1]-freqs[0]
     print "* Frequency resolution returned by mlab.psd:     ", f_resolution
     print "  1/(chunk_length_sec):                          ", 1/chunk_length_sec
     print "  These two have to agree."
     print "" # blank

     # Check the PSD through Parseval's Theorem
     rms_psd = np.sqrt(np.sum(pxx * f_resolution)) # Integrate PSD over all freqs

     rms = np.sqrt(np.mean(np.fabs(strain_chunk*my_window)**2)) # RMS of the windowed time series
     window_comp = np.sqrt(np.mean(my_window**2)) # Window compensation factor (in amplitude)

     print "* Confirm PSD calculation with Parseval's Theorem"
     print '  RMS (frequency domain):              ', rms_psd,         'Unit: strain_rms'
     print '  RMS (time domain with window comp.): ', rms/window_comp, 'Unit: strain_rms'
     print "  These two have to agree."
