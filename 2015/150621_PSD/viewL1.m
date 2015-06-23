
h5disp('L-L1_LOSC_4_V1-842657792-4096.hdf5')
% This is a strain time series
% Duration = 4096s
% Each strain data point is a GPS time starting at 842657792 with spacing
% of 0.000244s


% Read in strain values
strain = h5read('L-L1_LOSC_4_V1-842657792-4096.hdf5', '/strain/Strain')


%% Make time series plot at evenly spaces time intervals

% Pick a length of the 106777216 strain values available to plot that 
% avoids any NaN values
length = 10000000;
strain_to_plot = strain(1:length);

% Assign the starting time to be 0 and calculate all subsequent times
times = linspace(0, (length+1) / 4096, length);

% Plot the time series
plot(times, strain_to_plot)
title('Strain v. Time for L1 at 842657792');
xlabel('time (s)');
ylabel('strain');



%% Provide some statistical analysis of the data
standard_dev = std(strain_to_plot) % 1.4869e-17
avg = mean(strain_to_plot) % -2.0084e-20 -- this is almost 0

%% Plot a histogram of strain v. time data
histfit(strain_to_plot, 50)
xlabel('strain')
title('Histogram for Strain v. Time Data')


%% Try a different method of making the PSDs by adding arguments to pwelch
fs = 4096; % Hz - sampling frequency
tfft = 2; % FFT segment length - !!! I randomly chose a number
nfft = tfft*fs;
win = hann(nfft);
noverlap = nfft/2;

[psd_h, freq] = pwelch(strain_to_plot, win, noverlap, nfft, fs);
loglog(freq, sqrt(psd_h))
xlabel('Frequency (Hz)')
ylabel('Amplitude Spectral Density (strain/Sqrt[Hz])')
title('Sensitivity for L1 for 1 hour at GPS start time 842657792');
%axis([10 10000 0 inf]);

%% Make 100 PSDs to compare statistics at different frequency values
% !! - this section is under construction

% Split time series into num_segments number of evenly spaced chunks
num_segments = 1;
psd = [];
for index = 1:num_segments
    % Plot a PSD for each segment
    min_index = (index - 1) * (length/num_segments) + 1;
    max_index = min_index + length/num_segments - 1;
    
    % !!! - should use something other than pwelch to find PSD
    % maybe even make my own calculation of PSD
    current_psd = pwelch(strain_to_plot(min_index:max_index));
    psd = [psd current_psd];
    loglog(current_psd);
    hold on;
end

xlabel('Frequency (Hz)')
ylabel('No idea what this axis is')
title('PSDs for Equally Spaced Frequency Segments');
axis([10 10000 0 inf]);




%% Plot Gaussian for one frequency chunk

% Select one frequency
frequency = 1064; % Hz

psd_values = [];
for element = psd
    psd_values = [psd_values element(frequency)];
end
histfit(psd_values, 20)
title('PSDs at 1064Hz')
xlabel('Not sure what this axis is either')


%% Plug in sine wave to PSD and see what happens

