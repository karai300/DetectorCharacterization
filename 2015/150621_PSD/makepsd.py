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
	#plt.savefig('timeseries.pdf')


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
		#plt.savefig('original_PSD.pdf')
	return pxx, freqs



# Read in data
strain, time, channel_dictionary = rl.loaddata('L-L1_LOSC_4_V1-842657792-4096.hdf5')
time_spacing = time[1] - time[0]
fs = int(1/time_spacing)

# Select a data segment without errors ("good data")
seglist = rl.dq_channel_to_seglist(channel_dictionary['DEFAULT'], fs)
length = 16  # s
strain_seg = strain[seglist[0]][:(length * fs)]
time_seg = time[seglist[0]][:(length * fs)]

# Plot a time series
#plot_timeseries(time_seg, strain_seg)

# Calculate a PSD
pxx, freqs = plot_psd(time_seg, strain_seg, fs, True)

# Check the PSD through Parseval's Theorem
total_power = np.sum(pxx * (freqs[1] - freqs[0])) # Integrate PSD over all freqs
rms = np.sqrt(np.mean(np.fabs(strain_seg)**2))
print "\nConfirm PSD calculation with Parseval's Theorem"
print 'variance: ', rms**2
print 'total power: ', total_power, '\n'

# Plot several PSDs to compare statistics on each PSD

# Select data length based on closest power of two which will give 
# approximately the desired number of segments
num_segments = 63
approx_seg_length = len(strain_seg) / num_segments
seg_length = int(pow(2, np.ceil(np.log(approx_seg_length)/np.log(2))))
print 'seg length: ', seg_length
print 'strain length: ', len(strain_seg)

# !!! - Change NFFT based on number of segments. 
if num_segments <= 20:
	nfft = int(seg_length / 2)
	pad = nfft * 4
else:
	nfft = int(seg_length)
	pad = nfft * 2
	
#nfft = 4096
print 'nfft: ', nfft
# 4096 works for 1 seg, 10 segs, 20 segs
# 2048 looks ok for 20 segs but not great
# 2048 looks pretty good for 50 segs

psds = []
plt.figure(3)
max_index = -1
while (max_index + seg_length) < len(strain_seg):
	# Define indices for frequency chunks
	# !!! - I haven't used a while loop in ages, and this feels weird
	min_index = max_index + 1
	max_index = min_index + seg_length - 1

	# Calculate each PSD
	pxx, freqs = mlab.psd(
		strain_seg[min_index:max_index], Fs=fs, NFFT=nfft, 
		noverlap=nfft/2, pad_to=pad)
	plt.loglog(freqs, np.sqrt(np.abs(pxx)))

	# Store list of tuples for PSD values
	psds.append((freqs, np.sqrt(pxx)))

plt.grid('on')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain /  Sqrt(Hz))')
plt.title('Several PSDs for L1 data starting at GPS ' + str(time_seg[0]))
plt.ylim([1e-26, 1e-16])
#plt.savefig('manyPSDs.pdf')

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
plt.title('Average of several PSDs for L1 data starting at GPS ' + str(time_seg[0])) 
plt.ylim([1e-26, 1e-16])
plt.savefig('psd_statistics.pdf')

# Display histogram for a chosen frequency (in Hz as first parameter)
plot_histogram(500, psds)
print get_freq_statistics(freq, psds)
	

plt.show()




