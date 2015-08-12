from __future__ import division

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl


def plot_histogram(freq_selected, psds):
	
	"""
	Plot a histogram of various PSD values at one selected frequency.

	freq_selected -- selected frequency in Hz

	psds -- a dictionary with frequency values as the key and a list of PSD
	values as the corresponding dictionary value 

	"""

	# Store psd_h for selected frequency in each PSD
	psd_h = psds[freq_selected]	
	
	# Plot histogram
	#plt.figure(5)
	#bins = np.arange(min(psd_h), max(psd_h) + binwidth, binwidth)
	plt.hist(psd_h, 50, label=str(freq_selected) + 'Hz')
	plt.xlabel('PSD (strain^2 / Hz)')
	plt.ylabel('Count')
	plt.legend()
	#plt.savefig('histogram' + str(freq_selected) + '.png')
	

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
	


def plot_many_psds(
	seglist, seg_index, time, strain, fs, n_chunk, plotting=False):

	"""
	Calculate n_chunk PSDs for a given data segment.

	If n_chunk is too large for the segment, it will be reduced to the 
	maximum allowable value.

	seglist -- a list of segments of strain data without errors

	seg_index -- an index indicating which strain segment is used

	time -- array of all time values

	strain -- array of all strian values, corresponding to the time values

	fs -- sampling frequency in Hz

	n_chunk -- number of requested chunks to divide the timeseries into and
	calculate a PSD for
	
	plotting -- Boolean; if true plots are produced

	Returns:

	psds_dict -- a dictionary with a key of frequency and a value of a
	list of all associated PSD values 
	"""

	print '=== PSD statistics ==='
	print '* Use segment #', seg_index
	time_seg = time[seglist[seg_index]]
	time_seg_max = max(time_seg)
	time_seg_min = min(time_seg)
	segment_length = int((time_seg_max - time_seg_min) * fs)  # num samples
	print '* Length of the segment: ', time_seg_max - time_seg_min, 'sec =>', segment_length, ' samples'

	chunk_length = 2**12  # number of samples in a chunk
	chunk_length_sec = chunk_length / fs # chunk length in seconds
	print '* Length of one chunk: ', chunk_length_sec, ' sec => ', chunk_length, ' samples'

	# User selects desired number of chunks
	print '* Requested number of chunks: ', n_chunk 

	# Check that segment is of sufficient length
	analyze_length = (chunk_length/2) * (n_chunk + 1)
	if analyze_length > segment_length:
		print '  The requested data length exceeds the length of the segment.'
		n_chunk = int(segment_length/(chunk_length/2) - 1) 
		print '  Reduced the requested number of chunks: ', n_chunk
	print ''  # blank line

	# Calculate PSDs for each chunk
	if plotting:
		plt.figure(4)
	num_PSDs = 0
	psds = []
	strain_seg = strain[seglist[seg_index]]
	my_window = mlab.window_hanning(np.ones(chunk_length))
	for i_PSD in range(n_chunk):
		i_start = int(i_PSD * (chunk_length/2))
		i_end = int(i_start + chunk_length - 1)
		pxx, freqs = mlab.psd(
			strain_seg[i_start:i_end], Fs=fs, NFFT=chunk_length, 
			noverlap=chunk_length/2, window=my_window)
		if plotting:
			plt.loglog(freqs, np.sqrt(np.fabs(pxx)))		
		num_PSDs += 1
		
		# Store list of tuples for PSD values
		psds.append((freqs, np.fabs(pxx)))
	print '* Finished processing all chunks'	
	print '* Number of PSDs plotted: ', num_PSDs 
	print ''  # blank line


	psds_dict = {}
	for psd in psds:
		# Loop over all frequencies for a given PSD array
		for index, freq in enumerate(psd[0]):
			# If the frequency is not in dictionary, add to keys
			if freq not in psds_dict.keys():
				psds_dict[freq] = [psd[1][index]]
				
			# Append new PSD value if frequency already in dict
			else:
				psds_dict[freq].append(psd[1][index])


	if plotting:
		plt.grid('on')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('PSD (strain /  Sqrt(Hz))')
		plt.title(
			str(num_PSDs) + ' PSDs for L1 data starting at GPS ' + 
			str(time_seg[0]))
		plt.ylim([1e-26, 1e-16])
		#plt.savefig('manyPSDs.png')

	return psds_dict	


def plot_psd(time_seg, strain_seg, fs, chunk_length, window, plotting=False):
	
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

# Plot PSDs for 200 data chunks for the first good segment
psds = plot_many_psds(seglist, 0, time, strain, fs, 200)

# Calculate PSD for maximum number of chunks
psds_max = plot_many_psds(seglist, 0, time, strain, fs, 5)


# Calculate histograms for PSD distribution for many PSDs at different freqs
print '=== Generating Histograms ==='
plt.figure(5)
plot_histogram(100, psds)
plt.figure(6)
plot_histogram(500, psds)
plt.figure(7)
plot_histogram(1000, psds)
print '* Finished histogram generation'
print ''  # blank line

# Calculate P1 and P2 for a smaller data segment
print '=== Calculating P0, P1, and P2 ==='
# Create an array of tuples of frequency, c1, and c2 values
freqs = []
c1_values = []
c2_values = []
for freq in psds_max.keys():
	# Convert PSD lists to numpy arrays
	psds_max[freq] = np.asarray(psds_max[freq])
	psds[freq] = np.asarray(psds[freq])

	# Calculate powers
	P0 = np.mean(psds_max[freq])
	P1 = np.mean(psds[freq])
	P2 = np.mean((psds[freq])**2)	
	
	# Calculate c1 and c2
	c1_values.append(P1/P0 - 1)
	c2_values.append(0.5 * (P2 / (P1**2) - 2))
	freqs.append(freq)
psd_statistics = np.array(zip(freqs, c1_values, c2_values), dtype=[
	('freq', float),('c1', float),('c2', float)])
print '* Finished calculating PSD statistics'

# Plot c1 and c2 for various frequencies
plt.figure(8)
c1_plot = plt.subplot(211)
plt.plot(psd_statistics['freq'], psd_statistics['c1'], label='c1')
plt.title(r'$c_{1}$ and $c_{2}$ values for L1 data starting at GPS ' + str(time_seg[0]))
plt.legend()
plt.ylabel(r'$c_{1}$')
c1_plot.set_xscale('log')

c2_plot = plt.subplot(212, sharex=c1_plot)
plt.plot(psd_statistics['freq'], psd_statistics['c2'], label='c2')
plt.legend()
plt.ylabel(r'$c_{2}$')
plt.xlabel('Frequency (Hz)')
plt.savefig('Gaussianity.pdf')

plt.show()




