from __future__ import division

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl


def get_freq_statistics(freq_selected, psds):

        """
        Explore statistics for distribution of all PSDs at one frequency bin

        Given a set of PSDs computed from several frequency chunks of the 
        same data set, the spread of strain / sqrt(Hz) values are compared 
        for all PSDs at a selected frequency. Mean and standard deviation 
        are returned.
        
        freq_selected --  a frequency in Hz at which the PSDs will be compared

        psds -- a list of tuples, with one tuple representing the frequency
        and strain / sqrt(Hz) values for one PSD

        mean -- average of the strain / sqrt(Hz) a.k.a. psd_h values at 
        the selected frequency

        std -- standard deviation of the aforementioned values

        """

        # Store psd_h for selected frequncy in each PSD
        psd_h = []

        # Find corresponding PSD value for selected frequency
        for psd in psds:
                for index, freq in enumerate(psd[0]):
                        if freq == freq_selected:
                                psd_h.append(psd[1][index])
                                # !!! - awkward to have list of arrays


        # Calculate standard deviation and mean of corresponding psd_h
        mean = np.average(psd_h)
        std = np.std(psd_h)
        return mean, std


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
plt.figure(1)
plt.plot(time_seg - time_seg[0], strain_seg)
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')
plt.title('Time Series')


# Plot a PSD
pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=fs)
plt.figure(2)
plt.loglog(freqs, np.sqrt(pxx))
#plt.axis([10, 2000, 1e-46, 1e-36])
plt.grid('on')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain /  Sqrt(Hz))')
plt.title('PSD for L1 data starting at GPS ' + str(time_seg[0]))


# Plot several PSDs to compare statistics on each PSD
num_segments = 10
psds = []
plt.figure(3)
for index in range(num_segments):
	# Define indices for frequency chunks
	# !!! - I THINK THERE'S A BETTER WAY TO DO THIS
	min_index = int(index * len(strain_seg)/ num_segments)
	max_index = int(min_index + len(strain_seg)/num_segments)
	
	# Calculate each PSD
	pxx, freqs = mlab.psd(strain_seg[min_index:max_index], Fs=fs, NFFT=fs)
	plt.loglog(freqs, np.sqrt(pxx))

	# Store list of tuples for PSD values
	psds.append((freqs, np.sqrt(pxx)))

plt.grid('on')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain /  Sqrt(Hz))')
plt.title('Several PSDs for L1 data starting at GPS ' + str(time_seg[0]))


# Plot PSD statistics for every frequency in range
averages = []
stds = []
for freq in freqs:
	(mean, std) = get_freq_statistics(freq, psds)
	averages.append(mean)
	stds.append(std)
averages = np.asarray(averages)
stds = np.asarray(stds)

plt.figure(4)
plt.loglog(freqs, averages, lw=1, color='red', label='Mean')
plt.loglog(freqs, averages + 2*std, lw=1, color='blue', label='+2 sigma')
plt.loglog(freqs, averages - 2*std, lw=1, color='green', label='-2 sigma')
plt.grid('on')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (strain / Sqrt(Hz))')
plt.title('Average of several PSDs for L1 data starting at GPS ' + str(time_seg[0])) 


plt.show()




