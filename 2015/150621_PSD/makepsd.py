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

	binwidth -- desired width in PSD value of one histogram bin
	"""

	# Store psd_h for selected frequency in each PSD
	psd_h = []	

	# Find corresponding PSD value for each selected frequency
	for psd in psds:
		for index, freq in enumerate(psd[0]):
			if freq == freq_selected:
				psd_h.append(psd[1][index])

	
	# Plot histogram
	#plt.figure(5)
	#bins = np.arange(min(psd_h), max(psd_h) + binwidth, binwidth)
	plt.hist(psd_h, 50, label=str(freq_selected) + 'Hz')
	plt.xlabel('PSD (strain / (Sqrt(Hz))')
	plt.ylabel('Count')
	plt.legend()
	plt.savefig('histogram' + str(freq_selected) + '.pdf')
	

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
	#plt.savefig('timeseries.pdf')


def plot_psd(time_seg, strain_seg, fs, chunk_length, window, plotting=True):
	
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

	chunk_length -- the number of samples; timespan of segment multiplied
	by the sampling frequency

	window -- selected window for psd calculation; typically hanning

	plotting -- Boolean; if true plots are produced
	
	pxx -- PSD values corresponding to strain values

	freqs -- frequencies at which a PSD amplitude is calculated


	"""

	pxx, freqs = mlab.psd(
			      strain_seg, Fs=fs, NFFT=chunk_length, 
			      noverlap=chunk_length/2, window=window)
	if plotting:
		plt.figure(2)	
		plt.loglog(freqs, np.sqrt(pxx))
		plt.grid('on')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('PSD (strain /  Sqrt(Hz))')
		plt.title('PSD for L1 data starting at GPS ' + str(time_seg[0]))
		plt.ylim([1e-26, 1e-16])
		#plt.savefig('original_PSD.pdf')
	return pxx, freqs



# Read in data
fname = 'L-L1_LOSC_4_V1-842657792-4096.hdf5'
print '=== Start reading data ==='
print '* File name: ', fname
strain, time, channel_dictionary = rl.loaddata(fname)

data_len = len(time)
time_spacing = time[1] - time[0]
fs = int(1/time_spacing)

print '* Number of samples: ', data_len, ' = 2^', np.log2(data_len)
print '* Start GPS Time: ', time[0]
print '* End GPS Time: ', time[data_len - 1]
print '* Sampling Time: ', time_spacing, 'sec => ', fs, ' Hz'
print ''  # blank line

# Select a data segment without errors ("good data")
print '=== DQ good segments ==='
seglist = rl.dq_channel_to_seglist(channel_dictionary['DEFAULT'], fs)
num_seg = len(seglist)
print "* Number of uninterrupted 'good' segments: ", num_seg

seg_index = 0  # Select which good time segment to use
print '* Use segment #', seg_index
time_seg = time[seglist[seg_index]]
time_seg_min = min(time_seg)
time_seg_max = max(time_seg)
time_len = time_seg_max - time_seg_min
print '* Length of segment ', time_len, 'sec => ', time_len * fs, ' samples'
print ''

# Extract a chunk of data from the selected segment
print '=== Extracting a chunk of data from selected segment ==='
chunk_length_sec = 16  # s
chunk_length = chunk_length_sec * fs  # number of samples
print '* Chunk length: ', chunk_length_sec, 'sec => ', chunk_length, 'samples'

# Calculate PSD and confirm Parseval's Theorem for many data chunks
for chunk_index in range(10):

	print '* Chunk #', chunk_index
	strain_chunk = strain[seglist[seg_index]][
		chunk_length*chunk_index:chunk_length*(chunk_index+1)]
	time_chunk = time[seglist[seg_index]][
		chunk_length*chunk_index:chunk_length*(chunk_index+1)]

	print '* Extracted chunk length: ', len(time_chunk)
	print ''  # blank line

	# Plot a time series
	#plot_timeseries(time_seg, strain_seg)

	# Calculate a PSD
	my_window = mlab.window_hanning(np.ones(chunk_length))
	#pxx, freqs = mlab.psd(strain_chunk, Fs=fs, NFFT=chunk_length, 
			      #noverlap=chunk_length/2, window=my_window)
	pxx, freqs = plot_psd(time_chunk, strain_chunk, fs, chunk_length, 
			      my_window, False)
	
	f_resolution = freqs[1] - freqs[0]
	print '* Frequency resolution returned by mlab.psd: ', f_resolution,'Hz'
	print '  1/(chunk_length_sec): ', 1/chunk_length_sec, 'Hz'
	print '  This must agree'
	print ''  # blank line


	# Check the PSD through Parseval's Theorem
	rms_psd = np.sqrt(np.sum(pxx * f_resolution)) # Integrate PSD over all freqs
	rms = np.sqrt(np.mean(np.fabs(strain_chunk*my_window)**2)) # RMS of timeseries
	window_comp = np.sqrt(np.mean(my_window**2))  # Window compensation factor (in amplitude)
	print "* Confirm PSD calculation with Parseval's Theorem"
	print '  RMS (frequency domain): ', rms_psd, 'Unit: strain_rms'
	print '  RMS (time domain with window comp.: ', rms/window_comp, 'Unit: strain_rms'
	print '  This must agree'
	print ''

print '=== PSD statistics ==='
seg_index = 0
print '* Use segment #', seg_index
time_seg = time[seglist[seg_index]]
time_seg_max = max(time_seg)
time_seg_min = min(time_seg)
segment_length = int((time_seg_max - time_seg_min) * fs)  # number of samples
print '* Length of the segment: ', time_seg_max - time_seg_min, 'sec =>', segment_length, ' samples'

chunk_length = 2**12  # number of samples in a chunk
chunk_length_sec = chunk_length / fs # chunk length in seconds
print '* Length of one chunk: ', chunk_length_sec, ' sec => ', chunk_length, ' samples'

# User selects desired number of chunks
n_chunk_request = 200
print '* Requested number of chunks: ', n_chunk_request 

# Check that segment is of sufficient length
analyze_length = (chunk_length/2) * (n_chunk_request + 1)
if analyze_length > segment_length:
	print '  The requested data length exceeds the length of the segment.'
	n_chunk_request = int(segment_length/(chunk_length/2) - 1) 
	print '  Reduced the requested number of chunks: ', n_chunk_request
print ''  # blank line

# Calculate PSDs for each chunk
plt.figure(4)
num_PSDs = 0
psds = []
strain_seg = strain[seglist[seg_index]]
my_window = mlab.window_hanning(np.ones(chunk_length))
for i_PSD in range(n_chunk_request + 1):
	i_start = int(i_PSD * (chunk_length/2))
	i_end = int(i_start + chunk_length - 1)
	pxx, freqs = mlab.psd(
		strain_seg[i_start:i_end], Fs=fs, NFFT=chunk_length, 
		noverlap=chunk_length/2, window=my_window)
	plt.loglog(freqs, np.sqrt(np.fabs(pxx)))		
	num_PSDs += 1
	
	# Store list of tuples for PSD values
	psds.append((freqs, np.fabs(pxx)))
print '* Finished processing all chunks'	
print '* Number of PSDs plotted: ', num_PSDs 


plt.grid('on')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain /  Sqrt(Hz))')
plt.title(
	str(num_PSDs) + ' PSDs for L1 data starting at GPS ' + str(time_seg[0]))
plt.ylim([1e-26, 1e-16])
plt.savefig('manyPSDs.pdf')


# Calculate histograms for PSD distribution for many PSDs at different freqs
plt.figure(5)
binwidth = 1e-46
plot_histogram(100, psds)
plt.figure(6)
plot_histogram(500, psds)
plt.figure(7)
plot_histogram(1000, psds)

	
'''
# Plot PSD statistics for every frequency in range
many_mean = []
many_std = []
many_skew = []
many_kurt = []
for freq in freqs:
	mean, var, skewness, kurtosis = get_freq_statistics(freq, psds)
	many_mean.append(mean)
	many_std.append(np.sqrt(var))
	#if mean - 2*std < 0:
		#print freq, mean, mean - 2*std, std
many_mean = np.asarray(many_mean)
many_std = np.asarray(many_std)
many_skew = np.asarray(many_skew)
many_kurt = np.asarray(many_kurt)
plt.figure(4)
mean = plt.loglog(freqs, many_mean, lw=1, color='red')
std4 = plt.fill_between(
	freqs, many_mean, many_mean + 4*many_std, facecolor='yellow')
std2 = plt.fill_between(freqs, many_mean, many_mean + 2*many_std, facecolor='blue')
#plt.loglog(freqs, many_mean + 2*many_std, lw=1, color='blue', label='+2 sigma')
#plt.loglog(freqs, many_mean - 2*many_std, lw=1, color='green', label='-2 sigma')
plt.grid('on')
#plt.legend([mean, std4, std2], ['Mean', '4 std', '2std'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain / Sqrt(Hz))')
plt.title('Average of ' + str(num_PSDs) + ' PSDs for L1 data starting at GPS ' + str(time_seg[0])) 
plt.ylim([1e-26, 1e-16])
#plt.savefig('psd_statistics.pdf')

# Display histogram for a chosen frequency (in Hz as first parameter)
plot_histogram(500, psds)
print get_freq_statistics(freq, psds)
'''	

plt.show()




