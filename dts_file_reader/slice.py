# -*- coding: utf-8 -*-
#
#  DTS Slice data file(v4) reader
#
#  author: josef ling (jling@mrn.org)
#
#  Copyright (C) 2022  josef ling
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numpy import ndarray


class Channel:
    class Meta:
        def __init__(self):
            self.eu = None
            self.is_proportional_to_excitation = None
            self.factory_excitation_voltage = None
            self.serial_number = None
            self.delta_t = 0.0
            self.sample_rate_hz = 0
            self.number_of_samples = 0
            self.number_of_bits = 0

        def __repr__(self):
            return "Channel Meta:\n" \
                   "EU:{}\n" \
                   "ProportionalToExcitation:{}\n" \
                   "FactoryExcitationVoltage:{}\n" \
                   "SerialNumber:{}\n" \
                   "Delta_T:{}\n" \
                   "SampleRateHz:{}\n" \
                   "NumberOfSamples:{}\n" \
                   "NumberOfBits:{}".format(self.eu,
                                            self.is_proportional_to_excitation,
                                            self.factory_excitation_voltage,
                                            self.serial_number,
                                            self.delta_t,
                                            self.sample_rate_hz,
                                            self.number_of_samples,
                                            self.number_of_bits
                                            )

    class Summary:

        class Attribute:
            def __init__(self, value, unit: str):
                self.value = value
                self.unit = unit

        def __init__(self):
            self.peak_index = 0
            self.min_index = 0
            self.rise_start_index = 0
            self.rise_end_index = 0
            self.peak_vel = self.Attribute(None, 'rad/s')
            self.min_vel = self.Attribute(None, 'rad/s')
            self.time_to_peak = self.Attribute(None, 'ms')
            self.decel_time = self.Attribute(None, 'ms')
            self.fwhm = self.Attribute(None, 'ms')
            self.delta_t = self.Attribute(None, 'ms')
            self.rise_to_peak_slope = 0
            self.is_peak_user_selected = 0

        def __repr__(self):
            return "Channel Summary:\npeak_index:{}\nmin_index:{}\nrise_start_index:{}\nrise_end_index:{}\npeak_vel:{} {}\nfwhm:{} {}\nslope:{}\npeak_user_selected:{}".format(
                self.peak_index,
                self.min_index,
                self.rise_start_index,
                self.rise_end_index,
                self.peak_vel.value,
                self.peak_vel.unit,
                self.fwhm.value,
                self.fwhm.unit,
                self.rise_to_peak_slope,
                bool(self.is_peak_user_selected)
            )

    # channel constructor
    def __init__(self, number: int):
        self.number = number
        self.meta_data = self.Meta()
        self.summary_data = self.Summary()
        self.scaled_data = np.array([])

    def __repr__(self):
        return "\n\n+++++++++++++++++++\nChannel number:{}\n\n" \
               "{}\n\n" \
               "{}".format(self.number,
                           self.meta_data,
                           self.summary_data)

    def get_filtered_data(self, cfc: object = None, start: int = 0, stop: int = 0) -> ndarray:
        """
        Return a range of filtered single channel data or all of it

        BUTTERWORTH 4-POLE PHASELESS DIGITAL FILTER outlined in Appendix C of the
        SAE-J211 (revMARCH95) standard.
        <http://standards.sae.org/j211/1_201403/>

        :param cfc:
        :param stop:
        :param start:
        :rtype: object
        """

        from scipy.signal import filtfilt

        # if channel is proportional_to_excitation it is an accelerometer,
        # otherwise a velocelometer
        if cfc is None:
            cfc = 180 if self.meta_data.is_proportional_to_excitation else 1000

        # if no stop, all
        if stop == 0:
            stop = len(self.scaled_data)

        # prepare filter coefficients
        wd = 2 * np.pi * cfc * 2.0775
        T = 1 / self.meta_data.sample_rate_hz
        wa = np.sin(wd * T / 2) / np.cos(wd * T / 2)
        a0 = wa ** 2 / (1.0 + (2 ** 0.5) * wa + wa ** 2)
        a1 = 2 * a0
        a2 = a0
        b0 = 1
        b1 = -2 * (wa ** 2 - 1) / (1 + (2 ** 0.5) * wa + wa ** 2)
        b2 = (-1 + (2 ** 0.5) * wa - wa ** 2) / (1 + (2 ** 0.5) * wa + wa ** 2)
        coeff_b = np.array([a0, a1, a2])
        coeff_a = np.array([b0, -b1, -b2])
        return self.remove_baseline_offset(filtfilt(coeff_b, coeff_a,
                                                    self.scaled_data[start:stop],
                                                    axis=0, padtype=None, padlen=None, method='gust'
                                                    ))

    def remove_baseline_offset(self, single_channel_data, baseline_index_start=0, baseline_index_end=500) -> np.ndarray:
        """
        Remove baseline offset from a single channel array using the start and end to define location of baseline
        measure.
        @return: array of same size with baseline subtracted
        """

        if (not isinstance(baseline_index_start, int)) or (not isinstance(baseline_index_end, int)):
            raise ValueError("baseline_index_start and baseline_index_end values must be integers")

        if baseline_index_end < baseline_index_start:
            raise ValueError("baseline_index_end value must be greater than baseline_index_start")

        return single_channel_data - np.mean(single_channel_data[baseline_index_start:baseline_index_end])

    def get_channel_summary(self, method: str):
        """
        Summary parameters for channel data
        @return:
        """
        if self.summary_data.peak_index == 0:
            self.summary_data = get_data_summary(method,
                                                 self.meta_data.sample_rate_hz,
                                                 self.get_filtered_data()
                                                 )

        return self.summary_data


def get_data_summary(method: str, sample_rate_hz: int, data=None):
    """ For input data, return summary parameters based on methodology

        This method is not for channel data only, it can be used on resultant data (a derivative of multiple channels).
        That is why it is not a channel memember.
    """

    #######################################################################
    # initiate summary instance
    #######################################################################
    summary_data = Channel.Summary()

    #######################################################################
    # Find peak
    #######################################################################
    set_peak(summary_data, method, data)

    #######################################################################
    # Find rise start index
    #######################################################################
    set_rise_start(summary_data, data)

    #######################################################################
    # Find rise end index
    #######################################################################
    set_rise_end(summary_data, data)

    #######################################################################
    # delta_t
    #######################################################################
    set_delta_t(summary_data, sample_rate_hz)

    #######################################################################
    # fwhm
    #######################################################################
    set_fwhm(summary_data, sample_rate_hz, data)

    #######################################################################
    # time_to_peak
    #######################################################################
    set_time_to_peak(summary_data, sample_rate_hz)

    #######################################################################
    # decel_time
    #######################################################################
    set_decel_time(summary_data, sample_rate_hz)

    #######################################################################
    # peak_velocity
    #######################################################################
    set_peak_vel(summary_data, data)

    #######################################################################
    # slope of line fit to data between rise start and peak
    #######################################################################
    set_slope(summary_data, data, sample_rate_hz)

    return summary_data


def set_peak(summary_data: Channel.Summary, method: str, data=None):
    """
    Find peak
    """
    from scipy.signal import find_peaks

    if method == 'head':
        find_peaks_rounds = {
            '1': {'height': (100.0, None), 'rel_height': 0.5, 'threshold': (None, 1.0), 'width': (20, None)},
            '2': {'height': (100, None), 'threshold': (None, 0.3), 'width': (20, None)},
            '3': {'height': (50, None), 'threshold': (None, 0.3), 'width': (20, None)}
        }
    elif method == 'machine':
        # find_peaks_rounds = {
        #     '1': {'height': (100, None), 'rel_height': 0.5, 'threshold': (None, 1.0), 'width': (50, None), 'prominence': (2, 100)},
        #     '2': {'height': (100, None), 'threshold': (None, 0.3), 'width': (20, None)},
        #     '3': {'height': (20, None), 'threshold': (None, None), 'width': (7, None), 'prominence': (2, 250)}
        # }

        # this worked for all previous experiments(except 'legacy') until the Oct 2023 AE experiments
        # find_peaks_rounds = {
        #     '1': {'height': (100, 280), 'rel_height': 0.5, 'threshold': (None, 0.5), 'width': (50, None), 'prominence': (2, 15), 'distance': 25},
        #     '2': {'height': (100, 280), 'rel_height': 0.5, 'threshold': (None, 0.5), 'width': (None, None), 'prominence': (11, 15), 'distance': 25},
        #     '3': {'height': (100, 280), 'rel_height': 0.5, 'threshold': (None, 1.0), 'width': (30, 250), 'prominence': (10, 250), 'distance': 25},
        #     '4': {'height': (100, 280), 'rel_height': 0.5, 'threshold': (None, 1.0), 'width': (50, None), 'prominence': (2, 250), 'distance': 25},
        #     '5': {'height': (100, 280), 'threshold': (None, 0.3), 'width': (10, None), 'prominence': (2, 100), 'distance': 25}
        # }

        # tuned for Oct 2023 AE experiments
        # waiting for feedback from lab if any changes needed. This leaves more to manual selection which is maybe how it should be
        find_peaks_rounds = {
            '1': {'height': (150, 280), 'width': (None, None), 'rel_height': 0.1, 'threshold': (None, 2.0)}
        }
    else:
        raise RuntimeError('Method ' + method + ' is not valid.')

    # attempt some rounds of peak parameters
    for round_num, settings_str in find_peaks_rounds.items():
        # INFO
        #if method == 'machine':
        #    print(f"dts_file_reader.get_summary('{method}') : Peak identification attempt {round_num} : {settings_str}")
        # INFO

        peaks, peak_props = find_peaks(data, **settings_str)
        # INFO
        #if method == 'machine':
            #print(peak_props['peak_heights'], peak_props['widths'])
        #    print(peak_props)
        # INFO

        if len(peak_props['peak_heights']) != 0:
            # grab the highest peak of those returned
            summary_data.peak_index = peaks[np.argmax(peak_props['peak_heights'])]

            # of the available peaks, use the one that is widest
            #summary_data.peak_index = peaks[np.argmax(peak_props['widths'])]

            # min index just the lowest value after the peak index define above
            summary_data.min_index = np.argmin(data[summary_data.peak_index:]) + summary_data.peak_index
            break

    # if summary_data.peak_index == 0 after above attempts to identify peak,
    # just grab the max value
    if summary_data.peak_index == 0:
        # INFO
        print('no peak identified. dropping to max value')
        # INFO
        summary_data.peak_vel.value = np.max(data)
        summary_data.peak_index = np.argmax(data)
        summary_data.min_index = np.argmin(data)

    return summary_data


def set_rise_start(summary_data: Channel.Summary, data=None):
    """
    Find rise start index
    """
    # CHOP recommendation 2019: 5% of peak defines rise_start, rise_end, and delta_t
    # five_pct_of_peak_value = data[summary_data.peak_index] * 0.05
    # # take first occurance greater than zero
    # summary_data.rise_start_index = (np.where((data[:summary_data.peak_index] - five_pct_of_peak_value) > 0))[-1][0]

    # 5% peak for rise_start is too peak dependent, use 3 stdev instead
    three_stdev_cutoff = np.std(data[0:10000]) * 3

    # method 1
    # this method works backward from peak until it drops below 3 stdev
    # However, in one case i have seen a brief undershoot between the actual rise start and the ramp up
    # this method gets stuck in that dip and results in a delayed rise start placement
    # data_convolved = np.convolve(data[:summary_data.peak_index],
    #                              np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    #                              mode='same')
    # # three_stdev_cutoff_index = np.where(data[summary_data.peak_index::-1] < three_stdev_cutoff)[0][0]
    # three_stdev_cutoff_index = np.where(data_convolved[::-1] < three_stdev_cutoff)[0][0]
    # summary_data.rise_start_index = summary_data.peak_index - three_stdev_cutoff_index + 1

    # method 2
    # this method moves 25 ms before the peak index and works forward until it finds a single sample exceeding 3 stdev
    # because of noise, this has to be smoothed or it hits too early on noise
    data_convolved = np.convolve(data[(summary_data.peak_index - 1000):summary_data.peak_index],
                                 np.array([0.1, 0.1, 0.1, 0.1]),
                                 mode='same')

    # first capture data exceeding threshold
    exceed_three_stdev = np.where(data_convolved > three_stdev_cutoff)[0]
    # if we don't any return then the data is probably bad
    if exceed_three_stdev.shape[0] == 0:
        summary_data.rise_start_index = 0
        raise(Exception("Failed to identify rise_start_index"))
    else:
        three_stdev_cutoff_index = exceed_three_stdev[0]
        summary_data.rise_start_index = three_stdev_cutoff_index + summary_data.peak_index - 1000

    #print(summary_data.rise_start_index)
    # this can fail if no values returned
    #three_stdev_cutoff_index = np.where(data_convolved > three_stdev_cutoff)[0][0]
    #summary_data.rise_start_index = three_stdev_cutoff_index + summary_data.peak_index - 1000

    return summary_data


def set_rise_end(summary_data: Channel.Summary, data=None):
    """
    Find rise end index
    """
    # 5% peak for rise_start is too peak dependent, use 3 stdev instead
    three_stdev_cutoff = np.std(data[0:10000]) * 3

    summary_data.rise_end_index = (np.where(
        (data[summary_data.peak_index:summary_data.min_index] - three_stdev_cutoff) < 0) + summary_data.peak_index)[
        0, 0]

    a = data[summary_data.peak_index:summary_data.rise_end_index]
    signal_drop = np.where(np.logical_and(a < 0.5, a > -0.5)) + summary_data.peak_index
    is_signal_drop_around_rise_end = len(signal_drop[0]) > 40
    if is_signal_drop_around_rise_end:
        # print("signal drop detected")
        # get first point from left to right that crosses zero after subtracting 5% peak height
        # two known cases where signal cut out briefly before rise end. So, we change the search space.
        # instead of search from peak to trough as before, we find last point that is above 5% and add one for rise_end_index
        summary_data.rise_end_index = (np.where(
            (data[summary_data.peak_index:summary_data.min_index] - three_stdev_cutoff) > 0) + summary_data.peak_index)[
                                          0][-1] + 1

    return summary_data


def set_delta_t(summary_data: Channel.Summary, sample_rate_hz: int):
    """
    delta_t is time in milleseconds between rise_start and rise_end
    """
    summary_data.delta_t.value = (summary_data.rise_end_index - summary_data.rise_start_index) / (sample_rate_hz / 1000)
    return summary_data


def set_fwhm(summary_data: Channel.Summary, sample_rate_hz: int, data):
    """
    """
    fifty_pct_of_peak_value = data[summary_data.peak_index] * 0.50

    # new method look at line intersections
    idx = list(np.argwhere(np.diff(np.sign(np.full(len(data), fifty_pct_of_peak_value) - data))).flatten())
    # some presets
    fwhm_left_ips = idx[0]
    fwhm_right_ips = idx[-1]

    # filter out intersections to close together, possible spikes
    if len(idx) > 2:
        good = []
        dips = np.argwhere(np.diff(idx) > 40)
        dips = dips.squeeze().tolist()
        if isinstance(dips, list):
            for i in dips:
                good.append(idx[i + 1])
        else:
            good.append(idx[dips + 1])

        # finally, if a dip below 50% just to left, go back one intersection
        if data[good[-1] - 5] < fifty_pct_of_peak_value:
            fwhm_right_ips = good[-2]
        else:
            fwhm_right_ips = good[-1]

    summary_data.fwhm.value = (fwhm_right_ips - fwhm_left_ips) / (sample_rate_hz / 1000)

    return summary_data


def set_time_to_peak(summary_data: Channel.Summary, sample_rate_hz: int):
    summary_data.time_to_peak.value = (summary_data.peak_index - summary_data.rise_start_index) / (
                sample_rate_hz / 1000)
    return summary_data


def set_decel_time(summary_data: Channel.Summary, sample_rate_hz: int):
    summary_data.decel_time.value = (summary_data.rise_end_index - summary_data.peak_index) / (sample_rate_hz / 1000)
    return summary_data


def set_peak_vel(summary_data: Channel.Summary, data):
    summary_data.peak_vel.value = data[summary_data.peak_index]
    return summary_data


def set_slope(summary_data: Channel.Summary, data, sample_rate_hz: int):
    """
    Calculate slope of line fit to data between rise start and peak
    """

    # set x data as time steps
    x_data_ms = list(map(lambda x: x / (sample_rate_hz / 1000), range(0, data.shape[0])))

    # x as index steps only, not correct units
    # slope_calc_x = np.arange(summary_data.rise_start_index, summary_data.peak_index)
    # slope, intercept = np.polyfit(
    #     slope_calc_x,
    #     data[summary_data.rise_start_index:summary_data.peak_index],
    #     1
    # )

    # line fit for slope, intercept discarded
    slope, _ = np.polyfit(
        x_data_ms[summary_data.rise_start_index:summary_data.peak_index],
        data[summary_data.rise_start_index:summary_data.peak_index],
        1
    )

    # add slope to summary
    summary_data.rise_to_peak_slope = slope

    return summary_data


def set_user_selected_peak(summary_data: Channel.Summary, data, sample_rate_hz: int, user_selected_peak_index: int):
    summary_data.peak_index = user_selected_peak_index
    summary_data.is_peak_user_selected = 1
    summary_data = set_fwhm(summary_data, sample_rate_hz, data)
    summary_data = set_time_to_peak(summary_data, sample_rate_hz)
    summary_data = set_decel_time(summary_data, sample_rate_hz)
    summary_data = set_peak_vel(summary_data, data)
    summary_data = set_slope(summary_data, data, sample_rate_hz)

    return summary_data


def get_resultant(channels, channel_nums, start=0, stop=0):
    if stop == 0:
        stop = len(channels[channel_nums[0]].scaled_data)

    return np.sqrt(
        channels[channel_nums[0]].get_filtered_data(start=start, stop=stop) ** 2 +
        channels[channel_nums[1]].get_filtered_data(start=start, stop=stop) ** 2 +
        channels[channel_nums[2]].get_filtered_data(start=start, stop=stop) ** 2
    )


class Reader:
    """
    Parse DTS meta file with extension '*.dts'
    Selfishly only returning the metadata attributes i am interested in, not all that are available
    """

    def __init__(self):
        self.data_file_path = None
        self.data_directory = None
        self.data_file_stem = None

    def parse(self, i_file_path: str):
        """
        @type i_file_path: object
        """
        import xml.dom.minidom
        import struct

        try:
            from pathlib import Path
            self.data_file_path = Path(i_file_path)
            self.data_directory = self.data_file_path.parent
            self.data_file_stem = self.data_file_path.stem

            if self.data_file_path.suffix != '.dts':
                raise ValueError('Expected file extension ".dts" for input file.')

        except FileNotFoundError as e:
            raise e

        try:
            # return list of channels
            channels = []

            doc = xml.dom.minidom.parse(str(self.data_file_path))
            input_channels = doc.getElementsByTagName("AnalogInputChanel")

            for channel_num, input_channel in enumerate(input_channels):
                channel = Channel(channel_num)
                channel.meta_data.eu = input_channel.getAttribute("Eu")
                channel.meta_data.is_proportional_to_excitation = True if input_channel.getAttribute(
                    "ProportionalToExcitation") == 'True' else False
                channel.meta_data.factory_excitation_voltage = float(
                    input_channel.getAttribute("FactoryExcitationVoltage"))
                channel.meta_data.serial_number = input_channel.getAttribute("SerialNumber")

                channel_file_name = self.data_file_stem + '.' + str(channel_num) + '.chn'
                channel_file_path = self.data_directory / channel_file_name

                # parse the channel data file which contains a header and data
                with open(channel_file_path, 'rb') as f:

                    # file format magic key Slice channel file
                    # UInt32
                    byte_val = f.read(4)
                    magic_key = int.from_bytes(byte_val, "little", signed=True)
                    if magic_key != 741750047:
                        raise RuntimeError('Channel file corrupt or not Slice data channel file.')

                    # Offset (in bytes) from start of file to where data samples begin
                    # UInt64
                    f.seek(8, 0)
                    byte_val = f.read(8)
                    channel_data_start = int.from_bytes(byte_val, "little", signed=True)

                    # Number of samples in this file
                    # UInt64
                    byte_val = f.read(8)
                    number_of_samples = int.from_bytes(byte_val, "little", signed=True)
                    channel.meta_data.number_of_samples = int.from_bytes(byte_val, "little", signed=True)

                    # Number of bits per sample
                    # UInt32
                    byte_val = f.read(4)
                    number_of_bits = int.from_bytes(byte_val, "little", signed=True)
                    channel.meta_data.number_of_bits = number_of_bits

                    # 0 = Unsigned samples, 1 = signed samples
                    # UInt32
                    byte_val = f.read(4)
                    signed = int.from_bytes(byte_val, "little", signed=True)

                    # Sample rate (Hz)
                    # Double
                    f.seek(32, 0)
                    byte_val = f.read(8)
                    (sample_rate_hz) = int(struct.unpack('<d', byte_val)[0])
                    channel.meta_data.sample_rate_hz = int(sample_rate_hz)

                    # Time step between each sample. 'number_of_samples' is rounded here
                    # because it was 160002 and i want a nice clean timestep so i round to 160000
                    # at least one sample was before trigger in test files
                    # delta_t = sample_rate_hz / round(number_of_samples / 10) / 100000
                    channel.meta_data.delta_t = sample_rate_hz / round(number_of_samples / 10) / 100000

                    # Number of triggers. May be 0
                    # UInt16
                    byte_val = f.read(2)
                    number_of_triggers = int.from_bytes(byte_val, "little", signed=True)

                    # Trigger sample number. See slice ware file spec
                    N = number_of_triggers * 8

                    # Data-Zero level (in counts)
                    # Int32
                    f.seek(N + 70, 0)
                    byte_val = f.read(4)
                    data_zero_level_adc = int.from_bytes(byte_val, "little", signed=True)

                    # Scale factor MV (mV/Count)
                    # Double
                    f.seek((N + 74), 0)
                    byte_val = f.read(8)
                    scalefactor_mv = struct.unpack('<d', byte_val)[0]

                    # Scale factor EU
                    # mV/EU (non-proportional); mV/V/EU (proportional)
                    # Double
                    f.seek((N + 82), 0)
                    byte_val = f.read(8)
                    scalefactor_eu = struct.unpack('<d', byte_val)[0]

                    # go to where data samples begin
                    # read 16-bit ints
                    f.seek(channel_data_start, 0)
                    raw_data = np.zeros(number_of_samples, dtype=np.float64)
                    channel_data_indx = 0
                    while byte_val := f.read(int(number_of_bits / 8)):
                        raw_data[channel_data_indx] = int.from_bytes(byte_val, "little", signed=True)
                        channel_data_indx = channel_data_indx + 1

                    # scale raw data
                    # channel_data = (channel_data * scalefactorMV / scalefactorEU) - (dataZeroLevelADC / scalefactorEU * scalefactorMV)
                    # channel_data = (channel_data * scalefactor_mv / scalefactor_eu) - (data_zero_level_adc / scalefactor_eu * scalefactor_mv)
                    channel.scaled_data = (raw_data * float(scalefactor_mv) / float(scalefactor_eu)) - (
                            int(data_zero_level_adc) / float(scalefactor_eu) * float(scalefactor_mv))

                    # If units are deg/sec convert to radians
                    if channel.meta_data.eu.lower() == 'deg/sec':
                        # channel.scaled_data = np.multiply(channel.scaled_data, (np.pi / 180))
                        channel.scaled_data = np.multiply(channel.scaled_data, 0.017453292519943295)
                        channel.meta_data.eu = 'rad/s'
                    elif channel.meta_data.eu.lower() == 'g':
                        pass
                    else:
                        raise ValueError(f'Error in dts file.\nExpected channel {channel.meta_data.serial_number} Eu parameter to be "deg/sec" or "g".')

                    # For proportional channels adjust for excitation voltage
                    if channel.meta_data.is_proportional_to_excitation:
                        channel.scaled_data = np.divide(channel.scaled_data,
                                                        channel.meta_data.factory_excitation_voltage
                                                        )

                    # add channel to channel list to return
                    channels.append(channel)

            return channels

        except Exception as e:
            raise e
