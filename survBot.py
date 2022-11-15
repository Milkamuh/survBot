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
import matplotlib.pyplot as plt

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


def nsl_from_id(nwst_id):
    network, station, location = nwst_id.split('.')
    return dict(network=network, station=station, location=location)


def get_nwst_id(trace):
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
        self.plot_hour = self.starttime.hour
        self.current_day = self.starttime.julday
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
        self.html_fig_dir = 'figures'

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
        add_links = self.parameters.get('add_links')
        self.add_links = add_links if add_links else {}

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
                nwst_id = f'{nw}.{st}.'
                self.station_list.append(nwst_id)

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

    def read_data(self, re_read_at_hour=1, daily_overlap=2):
        '''
        read data method reads new data into self.stream

        :param re_read_at_hour: update archive at specified hour each day (hours up to 24)
        :param daily_overlap: re-read data of previous day until specified hour (hours up to 24)
        '''
        self.data = {}

        # re-read all data every new day
        curr_time = UTCDateTime()
        current_day = curr_time.julday
        current_hour = curr_time.hour
        yesterday = (curr_time - 24. * 3600.).julday
        if re_read_at_hour is not False and current_day != self.current_day and current_hour == re_read_at_hour:
            self.filenames_read = []
            self.dataStream = Stream()
            self.current_day = current_day

        # add all data to current stream
        for filename in self.filenames:
            if filename in self.filenames_read:
                continue
            try:
                st_new = read(filename)
                # add file to read filenames to prevent re-reading in case it is not the current day (or end of
                # previous day)
                if not filename.endswith(f'{current_day:03}') and not (
                        filename.endswith(f'{yesterday:03}') and current_hour <= daily_overlap):
                    self.filenames_read.append(filename)
            except Exception as e:
                print(f'Could not read file {filename}:', e)
                continue
            self.dataStream += st_new
        self.dataStream.merge()

        # organise data in dictionary with key for each station
        for trace in self.dataStream:
            nwst_id = get_nwst_id(trace)
            if not nwst_id in self.data.keys():
                self.data[nwst_id] = Stream()
            self.data[nwst_id].append(trace)

    def execute_qc(self):
        if self.reread_parameters:
            self.update_parameters()
        self.get_filenames()
        self.read_data()
        qc_starttime = UTCDateTime()

        self.analysis_print_list = []
        self.analysis_results = {}
        for nwst_id in sorted(self.station_list):
            stream = self.data.get(nwst_id)
            if stream:
                nsl = nsl_from_id(nwst_id)
                station_qc = StationQC(stream, nsl, self.parameters, self.keys, qc_starttime, self.verbosity,
                                       print_func=self.print)
                analysis_print_result = station_qc.return_print_analysis()
                station_dict = station_qc.return_analysis()
            else:
                analysis_print_result = self.get_no_data_station(nwst_id, to_print=True)
                station_dict = self.get_no_data_station(nwst_id)
            self.analysis_print_list.append(analysis_print_result)
            self.analysis_results[nwst_id] = station_dict

        self.update_status_message()
        return 'ok'

    def get_no_data_station(self, nwst_id, no_data='-', to_print=False):
        delay = self.get_station_delay(nwst_id)
        if not to_print:
            status_dict = {}
            for key in self.keys:
                if key == 'last active':
                    status_dict[key] = Status(message=timedelta(seconds=int(delay)), detailed_messages=['No data'])
                else:
                    status_dict[key] = Status(message=no_data, detailed_messages=['No data'])
            return status_dict
        else:
            items = [nwst_id.rstrip('.')] + [fancy_timestr(timedelta(seconds=int(delay)))]
            for _ in range(len(self.keys) - 1):
                items.append(no_data)
            return items

    def get_station_delay(self, nwst_id):
        """ try to get station delay from SDS archive using client"""
        locations = ['', '0', '00']
        channels = ['HHZ', 'HHE', 'HHN', 'VEI', 'EX1', 'EX2', 'EX3']
        network, station = nwst_id.split('.')[:2]

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
        first_exec = True
        status = 'ok'
        while status == 'ok' and self.refresh_period > 0:
            status = self.execute_qc()
            if self.outpath_html:
                self.write_html_table()
                if self.parameters.get('html_figures'):
                    self.write_html_figures(check_plot_time=not(first_exec))
            else:
                self.print_analysis()
            time.sleep(self.refresh_period)
            if not self.outpath_html:
                self.clear_prints()
            first_exec = False

    def console_print(self, itemlist, str_len=21, sep='|', seplen=3):
        assert len(sep) <= seplen, f'Make sure seperator has less than {seplen} characters'
        sl = sep.ljust(seplen)
        sr = sep.rjust(seplen)
        string = sl
        for item in itemlist:
            string += item.center(str_len) + sr
        self.print(string, flush=False)

    def check_plot_hour(self):
        ''' Check if new hour started '''
        current_hour = UTCDateTime().hour
        if not current_hour > self.plot_hour:
            return False
        if current_hour == 23:
            self.plot_hour = 0
        else:
            self.plot_hour += 1
        return True

    def get_fig_path_abs(self, nwst_id):
        return pjoin(self.outpath_html, self.get_fig_path_rel(nwst_id))

    def get_fig_path_rel(self, nwst_id, fig_format='png'):
        return os.path.join(self.html_fig_dir, f'{nwst_id.rstrip(".")}.{fig_format}')

    def check_fig_dir(self):
        fdir = pjoin(self.outpath_html, self.html_fig_dir)
        if not os.path.isdir(fdir):
            os.mkdir(fdir)

    def check_html_dir(self):
        if not os.path.isdir(self.outpath_html):
            os.mkdir(self.outpath_html)

    def write_html_figures(self, check_plot_time=True):
        """ Write figures for html, right now hardcoded hourly """
        if check_plot_time and not self.check_plot_hour():
            return
        self.check_fig_dir()

        for nwst_id in self.station_list:
            fig = plt.figure(figsize=(16, 9))
            fnout = self.get_fig_path_abs(nwst_id)
            st = self.data.get(nwst_id)
            if st:
                st.plot(fig=fig, show=False, draw=False, block=False, equal_scale=False, method='full')
                ax = fig.axes[0]
                ax.set_title(f'Hourly refreshed plot at (UTC) {UTCDateTime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                fig.savefig(fnout, dpi=150., bbox_inches='tight')
            plt.close(fig)

    def write_html_table(self, default_color='#e6e6e6'):
        self.check_html_dir()
        fnout = pjoin(self.outpath_html, 'survBot_out.html')
        if not fnout:
            return
        try:
            with open(fnout, 'w') as outfile:
                write_html_header(outfile, self.refresh_period)
                #write_html_table_title(outfile, self.parameters)
                init_html_table(outfile)

                # First write header items
                header = self.keys.copy()
                # add columns for additional links
                for key in self.add_links:
                    header.insert(-1, key)
                header_items = [dict(text='Station', color=default_color)]
                for check_key in header:
                    item = dict(text=check_key, color=default_color)
                    header_items.append(item)
                write_html_row(outfile, header_items, html_key='th')

                # Write all cells
                for nwst_id in self.station_list:
                    fig_name = self.get_fig_path_rel(nwst_id)
                    col_items = [dict(text=nwst_id.rstrip('.'), color=default_color, hyperlink=fig_name)]
                    for check_key in header:
                        if check_key in self.keys:
                            status_dict = self.analysis_results.get(nwst_id)
                            status = status_dict.get(check_key)
                            message, detailed_message = status.get_status_str()

                            # get background color
                            dt_thresh = [timedelta(seconds=sec) for sec in self.dt_thresh]
                            bg_color = get_bg_color(check_key, status, dt_thresh, hex=True)
                            if not bg_color:
                                bg_color = default_color

                            # add degree sign for temp
                            if check_key == 'temp':
                                if not type(message) in [str]:
                                    message = str(message) + deg_str

                            item = dict(text=str(message), tooltip=str(detailed_message), color=bg_color)
                        elif check_key in self.add_links:
                            value = self.add_links.get(check_key).get('URL')
                            link_text = self.add_links.get(check_key).get('text')
                            if not value:
                                continue
                            nw, st = nwst_id.split('.')[:2]
                            hyperlink_dict = dict(nw=nw, st=st, nwst_id=nwst_id)
                            link = value.format(**hyperlink_dict)
                            item = dict(text=link_text, tooltip=link, hyperlink=link, color=default_color)
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
        self.status_dict = {key: Status() for key in self.keys}
        self.activity_check()

        self.analyse_channels()

    def status_ok(self, key, detailed_message="Everything OK", status_message='OK', overwrite=False):
        current_status = self.status_dict.get(key)
        # do not overwrite existing warnings or errors
        if not overwrite and (current_status.is_warn or current_status.is_error):
            return
        self.status_dict[key] = StatusOK(message=status_message, detailed_messages=[detailed_message])

    def warn(self, key, detailed_message, last_occurrence=None, count=1):
        if key == 'other':
            self.status_other(detailed_message, last_occurrence, count)

        new_warn = StatusWarn(count=count, show_count=self.parameters.get('warn_count'))

        current_status = self.status_dict.get(key)

        # change this to something more useful, SMS/EMAIL/PUSH
        if self.verbosity:
            self.print(f'{UTCDateTime()}: {detailed_message}', flush=False)

        # if error, do not overwrite with warning
        if current_status.is_error:
            return

        if current_status.is_warn:
            current_status.count += count
        else:
            current_status = new_warn

        self._update_status(key, current_status, detailed_message, last_occurrence)

        # warnings.warn(message)

        # # update detailed status if already existing
        # current_message = self.detailed_status_dict.get(key)
        # current_message = '' if current_message in [None, '-'] else current_message + ' | '
        # self.detailed_status_dict[key] = current_message + detailed_message
        #
        # # this is becoming a little bit too complicated (adding warnings to existing)
        # current_status_message = self.status_dict.get(key)
        # current_status_message = '' if current_status_message in [None, 'OK', '-'] else current_status_message + ' | '
        # self.status_dict[key] = current_status_message + status_message

    def error(self, key, detailed_message, last_occurrence=None, count=1):
        new_error = StatusError(count=count, show_count=self.parameters.get('warn_count'))
        current_status = self.status_dict.get(key)
        if current_status.is_error:
            current_status.count += count
        else:
            current_status = new_error

        self._update_status(key, current_status, detailed_message, last_occurrence)

        # change this to something more useful, SMS/EMAIL/PUSH
        if self.verbosity:
            self.print(f'{UTCDateTime()}: {detailed_message}', flush=False)
        # warnings.warn(message)
        
    def status_other(self, detailed_message, status_message, last_occurrence=None, count=1):
        key = 'other'
        new_status = StatusOther(count=count, messages=[status_message])
        current_status = self.status_dict.get(key)
        if current_status.is_other:
            current_status.count += count
            current_status.messages.append(status_message)
        else:
            current_status = new_status

        self._update_status(key, current_status, detailed_message, last_occurrence)

    def _update_status(self, key, current_status, detailed_message, last_occurrence):
        current_status.detailed_messages.append(detailed_message)
        current_status.last_occurrence = last_occurrence

        self.status_dict[key] = current_status

    def activity_check(self):
        self.last_active = self.last_activity()
        if not self.last_active:
            status = StatusError()
        else:
            message = timedelta(seconds=int(self.program_starttime - self.last_active))
            status = Status(message=message)
        self.status_dict['last active'] = status

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
            status = self.status_dict[key]
            message = status.message
            if key == 'last active':
                items.append(fancy_timestr(message))
            elif key == 'temp':
                items.append(str(message) + deg_str)
            else:
                items.append(str(message))
        return items

    def return_analysis(self):
        return self.status_dict

    def get_last_occurrence_timestring(self, trace, indices):
        """ returns a nicely formatted string of the timedelta since program starttime and occurrence and abs time"""
        last_occur = self.get_last_occurrence(trace, indices)
        if not last_occur:
            return ''
        last_occur_dt = timedelta(seconds=int(self.program_starttime - last_occur))
        return f', Last occurrence: {last_occur_dt} ({last_occur.strftime("%Y-%m-%d %H:%M:%S")})'

    def get_last_occurrence(self, trace, indices):
        return self.get_time(trace, indices[-1])

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
            self.status_ok(key, detailed_message=f'U={(voltage[-1])}V')
            return

        n_overvolt = 0
        n_undervolt = 0

        warn_message = f'Trace {trace.get_id()}:'
        if len(overvolt) > 0:
            # try calculate number of voltage peaks from gaps between indices
            n_overvolt = len(np.where(np.diff(overvolt) > 1)[0]) + 1
            detailed_message = warn_message + f' {n_overvolt}x Voltage over {high_volt}V' \
                               + self.get_last_occurrence_timestring(trace, overvolt)
            self.warn(key, detailed_message=detailed_message, count=n_overvolt,
                      last_occurrence=self.get_last_occurrence(trace, overvolt))


        if len(undervolt) > 0:
            # try calculate number of voltage peaks from gaps between indices
            n_undervolt = len(np.where(np.diff(undervolt) > 1)[0]) + 1
            detailed_message = warn_message + f' {n_undervolt}x Voltage under {low_volt}V '\
                               + self.get_last_occurrence_timestring(trace, undervolt)
            self.warn(key, detailed_message=detailed_message, count=n_undervolt,
                      last_occurrence=self.get_last_occurrence(trace, undervolt))

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
                      detailed_message=f'Trace {trace.get_id()}: '
                                       f'Temperature over {max_temp}\N{DEGREE SIGN} at {trace.get_id()}!'
                                       + self.get_last_occurrence_timestring(trace, t_check),
                      last_occurrence=self.get_last_occurrence(trace, t_check))
        else:
            self.status_ok(key,
                           status_message=cur_temp,
                           detailed_message=f'Average temperature of last {dt_t_str}: {av_temp_str}')

    def pb_power_analysis(self, channel='EX2', pb_dict_key='pb_SOH2'):
        """ Analyse EX2 channel of PowBox """
        keys = ['230V', '12V']
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, keys)
        if not trace:
            return

        voltage = trace.data * 1e-6
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox 12V/230V check (EX2)', flush=False)
        voltage_check, voltage_dict, last_val = self.pb_voltage_ok(trace, voltage, pb_dict_key, channel=channel)

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
        if not trace:
            return

        voltage = trace.data * 1e-6
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox Router/Charger check (EX3)', flush=False)
        voltage_check, voltage_dict, last_val = self.pb_voltage_ok(trace, voltage, pb_dict_key, channel=channel)

        if voltage_check:
            for key in keys:
                self.status_ok(key)
            return

        soh3_params = self.parameters.get('POWBOX').get(pb_dict_key)
        self.in_depth_voltage_check(trace, voltage_dict, soh3_params, last_val)

    def in_depth_voltage_check(self, trace, voltage_dict, soh_params, last_val):
        """ Associate values in voltage_dict to error messages specified in SOH_params and warn."""
        for volt_lvl, ind_array in voltage_dict.items():
            if volt_lvl == 1:
                continue  # No need to do anything here
            if len(ind_array) > 0:
                # get result from parameter dictionary for voltage level
                result = soh_params.get(volt_lvl)
                for key, message in result.items():
                    # if result is OK, continue with next voltage level
                    if message == 'OK':
                        self.status_ok(key)
                        continue
                    if volt_lvl > 1:
                        # try calculate number of voltage peaks from gaps between indices
                        n_occurrences = len(np.where(np.diff(ind_array) > 1)[0]) + 1
                        self.warn(key=key,
                                  detailed_message=f'Trace {trace.get_id()}: '
                                          f'Found {n_occurrences} occurrence(s) of {volt_lvl}V: {key}: {message}'
                                                   + self.get_last_occurrence_timestring(trace, ind_array),
                                  count=n_occurrences,
                                  last_occurrence=self.get_last_occurrence(trace, ind_array))
                    # if last_val == current voltage (which is not 1) -> FAIL or last_val < 1: PBox no data
                    if volt_lvl == last_val or (volt_lvl == -1 and last_val < 1):
                        self.error(key, detailed_message=f'Last PowBox voltage state {last_val}V: {message}')

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

    def pb_voltage_ok(self, trace, voltage, pb_dict_key, channel=None):
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

        # Get voltage levels for classification
        voltage_dict = {}
        classified_indices = np.array([])

        # add classified levels to voltage_dict
        for volt in voltage_levels:
            indices = np.where((voltage < volt + pb_thresh) & (voltage > volt - pb_thresh))[0]
            voltage_dict[volt] = indices
            classified_indices = np.append(classified_indices, indices)

        # Warn in case of voltage under OK-level (1V)
        if len(under) > 0:
            # try calculate number of occurences from gaps between indices
            n_occurrences = len(np.where(np.diff(under) > 1)[0]) + 1
            voltage_dict[-1] = under
            self.status_other(detailed_message=f'Trace {trace.get_id()}: '
                              f'Voltage below {pb_ok}V in {len(under)} samples, {n_occurrences} time(s). '
                              f'Mean voltage: {np.mean(voltage):.2}'
                                       + self.get_last_occurrence_timestring(trace, under),
                              status_message='under 1V ({})'.format(n_occurrences))

        # classify last voltage values
        for volt in voltage_levels:
            if (last_voltage < volt + pb_thresh) and (last_voltage > volt - pb_thresh):
                last_val = volt
                break
        else:
            last_val = round(last_voltage, 2)

        # in case not all voltage values could be classified
        if not len(classified_indices) == len(voltage):
            all_indices = np.arange(len(voltage))
            unclassified_indices = all_indices[~np.isin(all_indices, classified_indices)]
            n_unclassified = len(unclassified_indices)
            max_uncl = self.parameters.get('THRESHOLDS').get('unclassified')
            if max_uncl and n_unclassified > max_uncl:
                self.status_other(detailed_message=f'Trace {trace.get_id()}: '
                                               f'{n_unclassified}/{len(all_indices)} '
                                               f'unclassified voltage values in channel {trace.get_id()}',
                                  status_message=f'{channel}: {n_unclassified} uncl.')

        return False, voltage_dict, last_val

    def get_time(self, trace, index):
        """ get UTCDateTime from trace and index"""
        return trace.stats.starttime + trace.stats.delta * index


class Status(object):
    def __init__(self, message='-', detailed_messages=None, count: int = 0, last_occurrence=None, show_count=True):
        if detailed_messages is None:
            detailed_messages = []
        self.show_count = show_count
        self.message = message
        self.messages = [message]
        self.detailed_messages = detailed_messages
        self.count = count
        self.last_occurrence = last_occurrence
        self.is_warn = None
        self.is_error = None
        self.is_other = False

    def set_warn(self):
        self.is_warn = True

    def set_error(self):
        self.is_warn = False
        self.is_error = True

    def set_ok(self):
        self.is_warn = False
        self.is_error = False

    def get_status_str(self):
        message = self.message
        if self.count > 1 and self.show_count:
            message += f' ({self.count})'
        detailed_message = ''

        for index, dm in enumerate(self.detailed_messages):
            if index > 0:
                detailed_message += ' | '
            detailed_message += dm

        return message, detailed_message


class StatusOK(Status):
    def __init__(self, message='OK', detailed_messages=None):
        super(StatusOK, self).__init__(message=message, detailed_messages=detailed_messages)
        self.set_ok()


class StatusWarn(Status):
    def __init__(self, message='WARN', count=1, last_occurence=None, detailed_messages=None, show_count=False):
        super(StatusWarn, self).__init__(message=message, count=count, last_occurrence=last_occurence,
                                         detailed_messages=detailed_messages, show_count=show_count)
        self.set_warn()


class StatusError(Status):
    def __init__(self, message='FAIL', count=1, last_occurence=None, detailed_messages=None, show_count=False):
        super(StatusError, self).__init__(message=message, count=count, last_occurrence=last_occurence,
                                          detailed_messages=detailed_messages, show_count=show_count)
        self.set_error()

    
class StatusOther(Status):
    def __init__(self, messages=None, count=1, last_occurence=None, detailed_messages=None):
        super(StatusOther, self).__init__(count=count, last_occurrence=last_occurence,
                                          detailed_messages=detailed_messages)
        if messages is None:
            messages = []
        self.messages = messages
        self.is_other = True
    
    def get_status_str(self):
        if self.messages == []:
            return '-'
    
        message = ''
        for index, mes in enumerate(self.messages):
            if index > 0:
                message += ' | '
            message += mes

        detailed_message = ''
        for index, dm in enumerate(self.detailed_messages):
            if index > 0:
                detailed_message += ' | '
            detailed_message += dm

        return message, detailed_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call survBot')
    parser.add_argument('-html', dest='html_path', default=None, help='filepath for HTML output')
    args = parser.parse_args()

    survBot = SurveillanceBot(parameter_path='parameters.yaml', outpath_html=args.html_path)
    survBot.start()
