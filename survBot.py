#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = '0.1'
__author__ = 'Marcel Paffrath'

import os
import traceback
import yaml
import argparse

import time
from datetime import timedelta
import numpy as np

from obspy import read, UTCDateTime, Stream
from obspy.clients.filesystem.sds import Client

from write_utils import write_html_text, write_html_row, write_html_footer, write_html_header, get_print_title_str,\
    init_html_table, finish_html_table
from utils import get_bg_color

pjoin = os.path.join
UP = "\x1B[{length}A"
CLR = "\x1B[0K"
deg_str = '\N{DEGREE SIGN}C'


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def nsl_from_id(st_id):
    network, station, location = st_id.split('.')
    return dict(network=network, station=station, location=location)


def get_st_id(trace):
    stats = trace.stats
    return f'{stats.network}.{stats.station}.'  # {stats.location}'


def fancy_timestr(dt, thresh=600, modif='+'):
    if dt > timedelta(seconds=thresh):
        value = f'{modif} ' + str(dt) + f' {modif}'
    else:
        value = str(dt)
    return value


class SurveillanceBot(object):
    def __init__(self, parameter_path, outpath_html=None):
        self.keys = ['last active', '230V', '12V', 'router', 'charger', 'voltage', 'temp', 'other']
        self.parameter_path = parameter_path
        self.update_parameters()
        self.starttime = UTCDateTime()
        self.outpath_html = outpath_html
        self.filenames = []
        self.filenames_read = []
        self.station_list = []
        self.analysis_print_list = []
        self.analysis_results = {}
        self.dataStream = Stream()
        self.data = {}
        self.print_count = 0
        self.status_message = ''

        self.cl = Client(self.parameters.get('datapath'))  # TODO: Check if this has to be loaded again on update
        self.get_stations()

    def update_parameters(self):
        self.parameters = read_yaml(self.parameter_path)
        self.reread_parameters = self.parameters.get('reread_parameters')
        self.dt_thresh = [int(val) for val in self.parameters.get('dt_thresh')]
        self.verbosity = self.parameters.get('verbosity')
        self.stations_blacklist = self.parameters.get('stations_blacklist')
        self.networks_blacklist = self.parameters.get('networks_blacklist')
        self.refresh_period = self.parameters.get('interval')
        self.transform_parameters()

    def transform_parameters(self):
        for key in ['networks', 'stations', 'locations', 'channels']:
            parameter = self.parameters.get(key)
            if type(parameter) == str:
                self.parameters[key] = list(self.parameters[key])
            elif type(parameter) not in [list]:
                raise TypeError(f'Bad input type for {key}: {type(key)}')

    def get_stations(self):
        networks = self.parameters.get('networks')
        stations = self.parameters.get('stations')

        self.station_list = []
        nwst_list = self.cl.get_all_stations()
        for nw, st in nwst_list:
            if self.stations_blacklist and st in self.stations_blacklist:
                continue
            if self.networks_blacklist and nw in self.networks_blacklist:
                continue
            if (networks == ['*'] or nw in networks) and (stations == ['*'] or st in stations):
                st_id = f'{nw}.{st}.'
                self.station_list.append(st_id)

    def get_filenames(self):
        self.filenames = []
        time_now = UTCDateTime()
        t1 = time_now - self.parameters.get('timespan') * 24 * 3600
        networks = self.parameters.get('networks')
        stations = self.parameters.get('stations')
        locations = self.parameters.get('locations')
        channels = self.parameters.get('channels')
        for network in networks:
            for station in stations:
                for location in locations:
                    for channel in channels:
                        self.filenames += list(self.cl._get_filenames(network, station, location, channel,
                                                                      starttime=t1, endtime=time_now))

    def read_data(self):
        self.data = {}

        # add all data to current stream
        for filename in self.filenames:
            if filename in self.filenames_read:
                continue
            try:
                st_new = read(filename)
                julday = UTCDateTime().julday
                # add file to read filenames to prevent re-reading in case it is not the current dayfile
                if not filename.endswith(str(julday)):
                    self.filenames_read.append(filename)
            except Exception as e:
                print(f'Could not read file {filename}:', e)
                continue
            self.dataStream += st_new
        self.dataStream.merge()

        # organise data in dictionary with key for each station
        for trace in self.dataStream:
            st_id = get_st_id(trace)
            if not st_id in self.data.keys():
                self.data[st_id] = Stream()
            self.data[st_id].append(trace)

    def execute_qc(self):
        if self.reread_parameters:
            self.update_parameters()
        self.get_filenames()
        self.read_data()
        qc_starttime = UTCDateTime()

        self.analysis_print_list = []
        self.analysis_results = {}
        for st_id in sorted(self.station_list):
            stream = self.data.get(st_id)
            if stream:
                nsl = nsl_from_id(st_id)
                station_qc = StationQC(stream, nsl, self.parameters, self.keys, qc_starttime, self.verbosity,
                                       print_func=self.print)
                analysis_print_result = station_qc.return_print_analysis()
                station_dict, warn_dict = station_qc.return_analysis()
            else:
                analysis_print_result = self.get_no_data_station(st_id, to_print=True)
                station_dict, warn_dict = self.get_no_data_station(st_id)
            self.analysis_print_list.append(analysis_print_result)
            self.analysis_results[st_id] = (station_dict, warn_dict)

        self.update_status_message()
        return 'ok'

    def get_no_data_station(self, st_id, no_data='-', to_print=False):
        delay = self.get_station_delay(st_id)
        if not to_print:
            status_dict = {}
            warn_dict = {}
            for key in self.keys:
                if key == 'last active':
                    status_dict[key] = timedelta(seconds=int(delay))
                    warn_dict[key] = 'No data within set timespan'
                else:
                    status_dict[key] = no_data
                    warn_dict[key] = 'No data'
            return status_dict, warn_dict
        else:
            items = [st_id.rstrip('.')] + [fancy_timestr(timedelta(seconds=int(delay)))]
            for _ in range(len(self.keys) - 1):
                items.append(no_data)
            return items

    def get_station_delay(self, st_id):
        """ try to get station delay from SDS archive using client"""
        locations = ['', '0', '00']
        channels = ['HHZ', 'HHE', 'HHN', 'VEI', 'EX1', 'EX2', 'EX3']
        network, station = st_id.split('.')[:2]

        times = []
        for channel in channels:
            for location in locations:
                t = self.cl.get_latency(network, station, location, channel)
                if t:
                    times.append(t)
        if len(times) > 0:
            return min(times)

    def print_analysis(self):
        self.print(200 * '+')
        title_str = get_print_title_str(self.parameters)
        self.print(title_str)
        if self.refresh_period > 0:
            self.print(f'Refreshing every {self.refresh_period}s.')
        items = ['Station'] + self.keys
        self.console_print(items, sep='---')
        for items in self.analysis_print_list:
            self.console_print(items)

    def start(self):
        '''
        Perform qc periodically.
        :param refresh_period: Update every x seconds
        :return:
        '''
        status = 'ok'
        while status == 'ok' and self.refresh_period > 0:
            status = self.execute_qc()
            if self.outpath_html:
                self.write_html_table()
            else:
                self.print_analysis()
            time.sleep(self.refresh_period)
            if not self.outpath_html:
                self.clear_prints()

    def console_print(self, itemlist, str_len=21, sep='|', seplen=3):
        assert len(sep) <= seplen, f'Make sure seperator has less than {seplen} characters'
        sl = sep.ljust(seplen)
        sr = sep.rjust(seplen)
        string = sl
        for item in itemlist:
            string += item.center(str_len) + sr
        self.print(string, flush=False)

    def write_html_table(self, default_color='#e6e6e6'):
        fnout = self.outpath_html
        if not fnout:
            return
        try:
            with open(fnout, 'w') as outfile:
                write_html_header(outfile, self.refresh_period)
                #write_html_table_title(outfile, self.parameters)
                init_html_table(outfile)

                # First write header items
                header_items = [dict(text='Station', color=default_color)]
                for check_key in self.keys:
                    item = dict(text=check_key, color=default_color)
                    header_items.append(item)
                write_html_row(outfile, header_items, html_key='th')

                # Write all cells
                for st_id in self.station_list:
                    col_items = [dict(text=st_id.rstrip('.'), color=default_color)]
                    for check_key in self.keys:
                        status_dict, detailed_dict = self.analysis_results.get(st_id)
                        status = status_dict.get(check_key)

                        # get background color
                        dt_thresh = [timedelta(seconds=sec) for sec in self.dt_thresh]
                        bg_color = get_bg_color(check_key, status, dt_thresh, hex=True)
                        if not bg_color:
                            bg_color = default_color

                        # add degree sign for temp
                        if check_key == 'temp':
                            if not type(status) in [str]:
                                status = str(status) + deg_str

                        item = dict(text=str(status), tooltip=str(detailed_dict.get(check_key)),
                                    color=bg_color)
                        col_items.append(item)
                    write_html_row(outfile, col_items)

                finish_html_table(outfile)
                write_html_text(outfile, self.status_message)
                write_html_footer(outfile)
        except Exception as e:
            print(f'Could not write HTML table to {fnout}:')
            print(traceback.format_exc())

    def update_status_message(self):
        timespan = timedelta(seconds=int(self.parameters.get('timespan') * 24 * 3600))
        self.status_message = f'Program starttime (UTC) {self.starttime.strftime("%Y-%m-%d %H:%M:%S")} | ' \
                              f'Current time (UTC) {UTCDateTime().strftime("%Y-%m-%d %H:%M:%S")} | ' \
                              f'Refresh period: {self.refresh_period}s | '\
                              f'Showing data of last {timespan}'

    def print(self, string, **kwargs):
        clear_end = CLR + '\n'
        n_nl = string.count('\n')
        string.replace('\n', clear_end)
        print(string, end=clear_end, **kwargs)
        self.print_count += n_nl + 1  # number of newlines + actual print with end='\n' (no check for kwargs end!)
        # print('pc:', self.print_count)

    def clear_prints(self):
        print(UP.format(length=self.print_count), end='')
        self.print_count = 0


class StationQC(object):
    def __init__(self, stream, nsl, parameters, keys, starttime, verbosity, print_func):
        """
        Station Quality Check class.
        :param nsl: dictionary containing network, station and location (key: str)
        :param parameters: parameters dictionary from parameters.yaml file
        """
        self.stream = stream
        self.nsl = nsl
        self.network = nsl.get('network')
        self.station = nsl.get('station')
        self.location = nsl.get('location')
        self.parameters = parameters
        self.program_starttime = starttime
        self.verbosity = verbosity
        self.last_active = False
        self.print = print_func

        timespan = self.parameters.get('timespan') * 24 * 3600
        self.analysis_starttime = self.program_starttime - timespan

        self.keys = keys
        self.detailed_status_dict = {key: None for key in self.keys}
        self.status_dict = {key: '-' for key in self.keys}
        self.activity_check()

        self.analyse_channels()

    def status_ok(self, key, message=None, status_message='OK'):
        self.status_dict[key] = status_message
        if message:
            self.detailed_status_dict[key] = message

    def warn(self, key, detailed_message, status_message='WARN'):
        # update detailed status if already existing
        current_message = self.detailed_status_dict.get(key)
        current_message = '' if current_message in [None, '-'] else current_message + ' | '
        self.detailed_status_dict[key] = current_message + detailed_message

        # this is becoming a little bit too complicated (adding warnings to existing)
        current_status_message = self.status_dict.get(key)
        current_status_message = '' if current_status_message in [None, 'OK', '-'] else current_status_message + ' | '
        self.status_dict[key] = current_status_message + status_message

        # change this to something more useful, SMS/EMAIL/PUSH
        if self.verbosity:
            self.print(f'{UTCDateTime()}: {detailed_message}', flush=False)
        # warnings.warn(message)

    def error(self, key, message):
        self.detailed_status_dict[key] = message
        self.status_dict[key] = 'FAIL'
        # change this to something more useful, SMS/EMAIL/PUSH
        if self.verbosity:
            self.print(f'{UTCDateTime()}: {message}', flush=False)
        # warnings.warn(message)

    def activity_check(self):
        self.last_active = self.last_activity()
        if not self.last_active:
            message = 'FAIL'
        else:
            message = timedelta(seconds=int(self.program_starttime - self.last_active))
        self.status_dict['last active'] = message

    def last_activity(self):
        if not self.stream:
            return
        endtimes = []
        for trace in self.stream:
            endtimes.append(trace.stats.endtime)
        if len(endtimes) > 0:
            return max(endtimes)

    def analyse_channels(self):
        if self.verbosity > 0:
            self.print(150 * '#')
            self.print('This is StationQT. Calculating quality for station'
                       ' {network}.{station}.{location}'.format(**self.nsl))
        self.voltage_analysis()
        self.pb_temp_analysis()
        self.pb_power_analysis()
        self.pb_rout_charge_analysis()

    def return_print_analysis(self):
        items = [f'{self.network}.{self.station}']
        for key in self.keys:
            item = self.status_dict[key]
            if key == 'last active':
                items.append(fancy_timestr(item))
            elif key == 'temp':
                items.append(str(item) + deg_str)
            else:
                items.append(str(item))
        return items

    def return_analysis(self):
        return self.status_dict, self.detailed_status_dict

    def get_last_occurrence_timestring(self, trace, indices):
        """ returns a nicely formatted string of the timedelta since program starttime and occurrence and abs time"""
        last_occur = self.get_time(trace, indices[-1])
        if not last_occur:
            return ''
        last_occur_dt = timedelta(seconds=int(self.program_starttime - last_occur))
        return f', Last occurrence: {last_occur_dt} ({last_occur.strftime("%Y-%m-%d %H:%M:%S")})'

    def voltage_analysis(self, channel='VEI'):
        """ Analyse voltage channel for over/undervoltage """
        key = 'voltage'
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, key)
        if not trace: return
        voltage = trace.data * 1e-3
        low_volt = self.parameters.get('THRESHOLDS').get('low_volt')
        high_volt = self.parameters.get('THRESHOLDS').get('high_volt')

        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing Voltage check', flush=False)

        overvolt = np.where(voltage > high_volt)[0]
        undervolt = np.where(voltage < low_volt)[0]

        if len(overvolt) == 0 and len(undervolt) == 0:
            self.status_ok(key, message=f'U={(voltage[-1])}V')
            return

        n_overvolt = 0
        n_undervolt = 0

        warn_message = f'Trace {trace.get_id()}:'
        if len(overvolt) > 0:
            # try calculate number of voltage peaks from gaps between indices
            n_overvolt = len(np.where(np.diff(overvolt) > 1)[0]) + 1
            warn_message += f' {n_overvolt}x Voltage over {high_volt}V' \
                            + self.get_last_occurrence_timestring(trace, overvolt)
        if len(undervolt) > 0:
            # try calculate number of voltage peaks from gaps between indices
            n_undervolt = len(np.where(np.diff(undervolt) > 1)[0]) + 1
            warn_message += f' {n_undervolt}x Voltage under {low_volt}V ' \
                            + self.get_last_occurrence_timestring(trace, undervolt)
        self.warn(key, detailed_message=warn_message, status_message='WARN ({})'.format(n_overvolt + n_undervolt))

    def pb_temp_analysis(self, channel='EX1'):
        """ Analyse PowBox temperature output. """
        key = 'temp'
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, key)
        if not trace: return
        voltage = trace.data * 1e-6
        thresholds = self.parameters.get('THRESHOLDS')
        temp = 20. * voltage - 20
        # average temp
        timespan = min([self.parameters.get('timespan') * 24 * 3600, int(len(temp) / trace.stats.sampling_rate)])
        nsamp_av = int(trace.stats.sampling_rate) * timespan
        av_temp_str = str(round(np.mean(temp[-nsamp_av:]), 1)) + deg_str
        # dt of average
        dt_t_str = str(timedelta(seconds=int(timespan)))
        # current temp
        cur_temp = round(temp[-1], 1)
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox temperature check (EX1)', flush=False)
            self.print(f'Average temperature at {np.mean(temp)}\N{DEGREE SIGN}', flush=False)
            self.print(f'Peak temperature at {max(temp)}\N{DEGREE SIGN}', flush=False)
            self.print(f'Min temperature at {min(temp)}\N{DEGREE SIGN}', flush=False)
        max_temp = thresholds.get('max_temp')
        t_check = np.where(temp > max_temp)[0]
        if len(t_check) > 0:
            self.warn(key=key,
                      status_message=cur_temp,
                      detailed_message=f'Trace {trace.get_id()}: '
                              f'Temperature over {max_temp}\N{DEGREE SIGN} at {trace.get_id()}!'
                                       + self.get_last_occurrence_timestring(trace, t_check))
        else:
            self.status_ok(key,
                           status_message=cur_temp,
                           message=f'Average temperature of last {dt_t_str}: {av_temp_str}')

    def pb_power_analysis(self, channel='EX2', pb_dict_key='pb_SOH2'):
        """ Analyse EX2 channel of PowBox """
        keys = ['230V', '12V']
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, keys)
        if not trace: return
        voltage = trace.data * 1e-6
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox 12V/230V check (EX2)', flush=False)
        voltage_check, voltage_dict, last_val = self.pb_voltage_ok(trace, voltage, pb_dict_key, channel=channel,
                                                                   warn_keys=keys)
        if voltage_check:
            for key in keys:
                self.status_ok(key)
            return

        soh2_params = self.parameters.get('POWBOX').get(pb_dict_key)
        self.in_depth_voltage_check(trace, voltage_dict, soh2_params, last_val)

    def pb_rout_charge_analysis(self, channel='EX3', pb_dict_key='pb_SOH3'):
        """ Analyse EX3 channel of PowBox """
        keys = ['router', 'charger']
        pb_thresh = self.parameters.get('THRESHOLDS').get('pb_1v')
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, keys)
        if not trace: return

        voltage = trace.data * 1e-6
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox Router/Charger check (EX3)', flush=False)
        voltage_check, voltage_dict, last_val = self.pb_voltage_ok(trace, voltage, pb_dict_key, channel=channel,
                                                                   warn_keys=keys)
        if voltage_check:
            for key in keys:
                self.status_ok(key)
            return

        soh3_params = self.parameters.get('POWBOX').get(pb_dict_key)
        self.in_depth_voltage_check(trace, voltage_dict, soh3_params, last_val)

    def in_depth_voltage_check(self, trace, voltage_dict, soh_params, last_val):
        """ Associate values in voltage_dict to error messages specified in SOH_params and warn."""
        for volt_lvl, ind_array in voltage_dict.items():
            if volt_lvl == 1: continue  # No need to do anything here
            if len(ind_array) > 0:
                result = soh_params.get(volt_lvl)
                for key, message in result.items():
                    if message == 'OK':
                        self.status_ok(key)
                        continue
                    # try calculate number of voltage peaks from gaps between indices
                    n_occurrences = len(np.where(np.diff(ind_array) > 1)[0]) + 1
                    self.warn(key=key,
                              detailed_message=f'Trace {trace.get_id()}: '
                                      f'Found {n_occurrences} occurrence(s) of {volt_lvl}V: {key}: {message}'
                                               + self.get_last_occurrence_timestring(trace, ind_array),
                              status_message='WARN ({})'.format(n_occurrences))
                    if last_val != 1:
                        self.error(key, message=f'Last PowBox voltage state {last_val}V: {message}')

    def get_trace(self, stream, keys):
        if not type(keys) == list:
            keys = [keys]
        if len(stream) == 0:
            for key in keys:
                self.warn(key, 'NO DATA', 'NO DATA')
            return
        if len(stream) > 1:
            raise Exception('Ambiguity error')
        trace = stream[0]
        if trace.stats.endtime < self.analysis_starttime:
            for key in keys:
                self.warn(key, 'NO DATA', 'NO DATA')
            return
        return trace

    def pb_voltage_ok(self, trace, voltage, pb_dict_key, warn_keys, channel=None):
        """
        Checks if voltage level is ok everywhere and returns True. If it is not okay it returns a dictionary
        with each voltage value associated to the different steps specified in POWBOX > pb_steps. Also raises
        self.warn in case there are unassociated voltage values recorded.
        """
        pb_thresh = self.parameters.get('THRESHOLDS').get('pb_thresh')
        pb_ok = self.parameters.get('POWBOX').get('pb_ok')
        # possible voltage levels are keys of pb voltage level dict
        voltage_levels = list(self.parameters.get('POWBOX').get(pb_dict_key).keys())

        # get mean voltage value of last samples
        last_voltage = np.nanmean(voltage[-3:])

        # check if voltage is over or under OK-level (1V), if not return True
        over = np.where(voltage > pb_ok + pb_thresh)[0]
        under = np.where(voltage < pb_ok - pb_thresh)[0]
        if len(over) == 0 and len(under) == 0:
            return True, {}, last_voltage

        # Warn in case of voltage under OK-level (1V)
        if len(under) > 0:
            # try calculate number of occurences from gaps between indices
            n_occurrences = len(np.where(np.diff(under) > 1)[0]) + 1
            for key in warn_keys:
                self.warn(key=key,
                          detailed_message=f'Trace {trace.get_id()}: '
                                  f'Voltage below {pb_ok}V in {len(under)} samples, {n_occurrences} time(s). '
                                  f'Mean voltage: {np.mean(voltage):.2}'
                                           + self.get_last_occurrence_timestring(trace, under),
                          status_message='WARN ({})'.format(n_occurrences))

        # Get voltage levels for classification
        voltage_dict = {}
        classified_indices = np.array([])

        # add classified levels to voltage_dict
        for volt in voltage_levels:
            indices = np.where((voltage < volt + pb_thresh) & (voltage > volt - pb_thresh))[0]
            voltage_dict[volt] = indices
            classified_indices = np.append(classified_indices, indices)

        # classify last voltage values
        for volt in voltage_levels:
            if (last_voltage < volt + pb_thresh) and (last_voltage > volt - pb_thresh):
                last_val = volt
                break
        else:
            last_val = np.nan

        # in case not all voltage values could be classified
        if not len(classified_indices) == len(voltage):
            all_indices = np.arange(len(voltage))
            unclassified_indices = all_indices[~np.isin(all_indices, classified_indices)]
            n_unclassified = len(unclassified_indices)
            max_uncl = self.parameters.get('THRESHOLDS').get('unclassified')
            if max_uncl and n_unclassified > max_uncl:
                self.warn(key='other', detailed_message=f'Trace {trace.get_id()}: '
                                               f'{n_unclassified}/{len(all_indices)} '
                                               f'unclassified voltage values in channel {trace.get_id()}',
                          status_message=f'{channel}: {n_unclassified} uncl.')

        return False, voltage_dict, last_val

    def get_time(self, trace, index):
        """ get UTCDateTime from trace and index"""
        return trace.stats.starttime + trace.stats.delta * index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call survBot')
    parser.add_argument('-html', dest='html_filename', default=None, help='filename for HTML output')
    args = parser.parse_args()

    survBot = SurveillanceBot(parameter_path='parameters.yaml', outpath_html=args.html_filename)
    survBot.start()
