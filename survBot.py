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

from write_utils import write_html_text, write_html_row, write_html_footer, write_html_header, get_print_title_str, \
    init_html_table, finish_html_table
from utils import get_bg_color, modify_stream_for_plot, set_axis_yticks, set_axis_color, plot_axis_thresholds

try:
    import smtplib
    from email.mime.text import MIMEText

    mail_functionality = True
except ImportError:
    print('Could not import smtplib or mail. Disabled sending mails.')
    mail_functionality = False

pjoin = os.path.join
UP = "\x1B[{length}A"
CLR = "\x1B[0K"
deg_str = '\N{DEGREE SIGN}C'


def read_yaml(file_path, n_read=3):
    for index in range(n_read):
        try:
            with open(file_path, "r") as f:
                params = yaml.safe_load(f)
        except Exception as e:
            print(f'Could not read parameters file: {e}.\nWill try again {n_read - index - 1} time(s).')
            time.sleep(10)
            continue
        return params


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
        self.keys = ['last active', '230V', '12V', 'router', 'charger', 'voltage', 'mass', 'clock', 'temp', 'other']
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
        self.status_track = {}
        self.dataStream = Stream()
        self.data = {}
        self.print_count = 0
        self.status_message = ''
        self.html_fig_dir = 'figures'

        self.cl = Client(self.parameters.get('datapath'))  # TODO: Check if this has to be loaded again on update
        self.get_stations()

    def update_parameters(self):
        self.parameters = read_yaml(self.parameter_path)
        # add channels to list in parameters dicitonary
        self.parameters['channels'] = list(self.parameters.get('CHANNELS').keys())
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
                st_new = read(filename, dtype=float)
                # add file to read filenames to prevent re-reading in case it is not the current day (or end of
                # previous day)
                if not filename.endswith(f'{current_day:03}') and not (
                        filename.endswith(f'{yesterday:03}') and current_hour <= daily_overlap):
                    self.filenames_read.append(filename)
            except Exception as e:
                print(f'Could not read file {filename}:', e)
                continue
            self.dataStream += st_new
        self.dataStream.merge(fill_value=np.nan)

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
                station_qc = StationQC(self, stream, nsl, self.parameters, self.keys, qc_starttime,
                                       self.verbosity, print_func=self.print,
                                       status_track=self.status_track.get(nwst_id))
                analysis_print_result = station_qc.return_print_analysis()
                station_dict = station_qc.return_analysis()
            else:
                analysis_print_result = self.get_no_data_station(nwst_id, to_print=True)
                station_dict = self.get_no_data_station(nwst_id)
            self.analysis_print_list.append(analysis_print_result)
            self.analysis_results[nwst_id] = station_dict
        self.track_status()

        self.update_status_message()
        return 'ok'

    def track_status(self):
        """
        tracks error status of the last n_track + 1 errors.
        """
        n_track = self.parameters.get('n_track')
        if not n_track or n_track < 1:
            return
        for nwst_id, analysis_dict in self.analysis_results.items():
            if not nwst_id in self.status_track.keys():
                self.status_track[nwst_id] = {}
            for key, status in analysis_dict.items():
                if not key in self.status_track[nwst_id].keys():
                    self.status_track[nwst_id][key] = []
                track_lst = self.status_track[nwst_id][key]
                # pop list until length is n_track + 1
                while len(track_lst) > n_track:
                    track_lst.pop(0)
                track_lst.append(status.is_error)

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
        channels = ['HHZ', 'HHE', 'HHN'] + self.parameters.get('channels')
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
                    self.write_html_figures(check_plot_time=not (first_exec))
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
        """ Write figures for html (e.g. hourly) """
        if check_plot_time and not self.check_plot_hour():
            return

        for nwst_id in self.station_list:
            self.write_html_figure(nwst_id)

    def write_html_figure(self, nwst_id):
        """ Write figure for html for specified station """
        self.check_fig_dir()

        fig = plt.figure(figsize=(16, 9))
        fnout = self.get_fig_path_abs(nwst_id)
        st = self.data.get(nwst_id)
        if st:
            # TODO: this section might fail, adding try-except block for analysis and to prevent program from crashing
            try:
                st = modify_stream_for_plot(st, parameters=self.parameters)
                st.plot(fig=fig, show=False, draw=False, block=False, equal_scale=False, method='full')
                # set_axis_ylabels(fig, self.parameters, self.verbosity)
                set_axis_yticks(fig, self.parameters, self.verbosity)
                set_axis_color(fig)
                plot_axis_thresholds(fig, self.parameters, self.verbosity)
            except Exception as e:
                print(f'Could not generate plot for {nwst_id}:')
                print(traceback.format_exc())
            if len(fig.axes) > 0:
                ax = fig.axes[0]
                ax.set_title(f'Plot refreshed at (UTC) {UTCDateTime.now().strftime("%Y-%m-%d %H:%M:%S")}. '
                             f'Refreshed hourly or on FAIL status.')
                for ax in fig.axes:
                    ax.grid(True, alpha=0.1)
                fig.savefig(fnout, dpi=150., bbox_inches='tight')
        plt.close(fig)

    def write_html_table(self, default_color='#e6e6e6', default_header_color='#999', hide_keys_mobile=('other')):

        def get_html_class(status=None, check_key=None):
            """ helper function for html class if a certain condition is fulfilled """
            html_class = None
            if status and status.is_active:
                html_class = 'blink-bg'
            if check_key in hide_keys_mobile:
                html_class = 'hidden-mobile'
            return html_class

        self.check_html_dir()
        fnout = pjoin(self.outpath_html, 'survBot_out.html')
        if not fnout:
            return
        try:
            with open(fnout, 'w') as outfile:
                write_html_header(outfile, self.refresh_period)
                # write_html_table_title(outfile, self.parameters)
                init_html_table(outfile)

                # First write header items
                header = self.keys.copy()
                # add columns for additional links
                for key in self.add_links:
                    header.insert(-1, key)
                header_items = [dict(text='Station', color=default_header_color)]
                for check_key in header:
                    html_class = get_html_class(check_key=check_key)
                    item = dict(text=check_key, color=default_header_color, html_class=html_class)
                    header_items.append(item)
                write_html_row(outfile, header_items, html_key='th')

                # Write all cells
                for nwst_id in self.station_list:
                    fig_name = self.get_fig_path_rel(nwst_id)
                    nwst_id_str = nwst_id.rstrip('.')
                    col_items = [dict(text=nwst_id_str, color=default_color, hyperlink=fig_name,
                                      bold=True, tooltip=f'Show plot of {nwst_id_str}')]
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

                            html_class = get_html_class(status=status, check_key=check_key)
                            item = dict(text=str(message), tooltip=str(detailed_message), color=bg_color,
                                        html_class=html_class)
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

        if self.verbosity:
            print(f'Wrote html table to {fnout}')

    def update_status_message(self):
        timespan = timedelta(seconds=int(self.parameters.get('timespan') * 24 * 3600))
        self.status_message = f'Program starttime (UTC) {self.starttime.strftime("%Y-%m-%d %H:%M:%S")} | ' \
                              f'Current time (UTC) {UTCDateTime().strftime("%Y-%m-%d %H:%M:%S")} | ' \
                              f'Refresh period: {self.refresh_period}s | ' \
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
    def __init__(self, parent, stream, nsl, parameters, keys, starttime, verbosity, print_func, status_track={}):
        """
        Station Quality Check class.
        :param nsl: dictionary containing network, station and location (key: str)
        :param parameters: parameters dictionary from parameters.yaml file
        """
        self.parent = parent
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

        self.keys = keys
        self.status_dict = {key: Status() for key in self.keys}

        if not status_track:
            status_track = {}
        self.status_track = status_track

        self.start()

    @property
    def nwst_id(self):
        return f'{self.network}.{self.station}'

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
            # if error is new and not on program-startup set active and refresh plot (using parent class)
            if self.search_previous_errors(key, n_errors=1) is True:
                self.parent.write_html_figure(self.nwst_id)

        if self.verbosity:
            self.print(f'{UTCDateTime()}: {detailed_message}', flush=False)

        # do not send error mail if this is the first run (e.g. program startup) or state was already error (unchanged)
        if self.search_previous_errors(key) is True:
            self.send_mail(key, status_type='FAIL', additional_message=detailed_message)
            # set status to "inactive" after sending info mail
            current_status.is_active = False
        elif self.search_previous_errors(key) == 'active':
            current_status.is_active = True

        self._update_status(key, current_status, detailed_message, last_occurrence)

    def search_previous_errors(self, key, n_errors=None):
        """
        Check n_track + 1 previous statuses for errors.
        If first item in list is no error but all others are return True
        (first time n_track errors appeared if ALL n_track + 1 are error: error is old)
        If last item is error but not all items are error yet return keyword 'active' -> error active, no message sent
        In all other cases return False.
        This also prevents sending status (e.g. mail) in case of program startup
        """
        if n_errors is None:
            n_errors = self.parameters.get('n_track')

        # +1 to check whether n_errors +1 was no error (error is new)
        n_errors += 1

        previous_errors = self.status_track.get(key)
        # only if error list is filled n_track times
        if previous_errors and len(previous_errors) == n_errors:
            # if first entry was no error but all others are, return True (-> new Fail n_track times)
            if not previous_errors[0] and all(previous_errors[1:]):
                return True
        # in case previous_errors exists, last item is error but not all items are error, error still active
        elif previous_errors and previous_errors[-1] and not all(previous_errors):
            return 'active'
        return False

    def send_mail(self, key, status_type, additional_message=''):
        """ Send info mail using parameters specified in parameters file """
        if not mail_functionality:
            if self.verbosity:
                print('Mail functionality disabled. Return')
            return

        mail_params = self.parameters.get('EMAIL')
        if not mail_params:
            if self.verbosity:
                print('parameter "EMAIL" not set in parameter file. Return')
            return

        stations_blacklist = mail_params.get('stations_blacklist')
        if stations_blacklist and self.station in stations_blacklist:
            if self.verbosity:
                print(f'Station {self.station} listed in blacklist. Return')
            return

        networks_blacklist = mail_params.get('networks_blacklist')
        if networks_blacklist and self.network in networks_blacklist:
            if self.verbosity:
                print(f'Station {self.station} of network {self.network} listed in blacklist. Return')
            return

        sender = mail_params.get('sender')
        addresses = mail_params.get('addresses')
        server = mail_params.get('mailserver')
        if not sender or not addresses:
            if self.verbosity:
                print('Mail sender or addresses not correctly defined. Return')
            return
        dt = self.get_dt_for_action()
        text = f'{key}: Status {status_type} longer than {dt}: ' + additional_message
        msg = MIMEText(text)
        msg['Subject'] = f'new message on station {self.nwst_id}'
        msg['From'] = sender
        msg['To'] = ', '.join(addresses)

        # send message via SMTP server
        s = smtplib.SMTP(server)
        s.sendmail(sender, addresses, msg.as_string())
        s.quit()

    def get_dt_for_action(self):
        n_track = self.parameters.get('n_track')
        interval = self.parameters.get('interval')
        dt = timedelta(seconds=n_track * interval)
        return dt

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

    def activity_check(self, key='last_active'):
        self.last_active = self.last_activity()
        if not self.last_active:
            status = StatusError()
        else:
            dt_active = timedelta(seconds=int(self.program_starttime - self.last_active))
            status = Status(message=dt_active)
            self.check_for_inactive_message(key, dt_active)

        self.status_dict['last active'] = status

    def last_activity(self):
        if not self.stream:
            return
        endtimes = []
        for trace in self.stream:
            endtimes.append(trace.stats.endtime)
        if len(endtimes) > 0:
            return max(endtimes)

    def check_for_inactive_message(self, key, dt_active):
        dt_action = self.get_dt_for_action()
        interval = self.parameters.get('interval')
        if dt_action <= dt_active < dt_action + timedelta(seconds=interval):
            self.send_mail(key, status_type='Inactive')

    def start(self):
        self.analyse_channels()

    def analyse_channels(self):
        timespan = self.parameters.get('timespan') * 24 * 3600
        self.analysis_starttime = self.program_starttime - timespan

        if self.verbosity > 0:
            self.print(150 * '#')
            self.print('This is StationQT. Calculating quality for station'
                       ' {network}.{station}.{location}'.format(**self.nsl))
        self.activity_check()
        self.voltage_analysis()
        self.pb_temp_analysis()
        self.pb_power_analysis()
        self.pb_rout_charge_analysis()
        self.mass_analysis()
        self.clock_quality_analysis()

    def return_print_analysis(self):
        items = [self.nwst_id]
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

    def clock_quality_analysis(self, channel='LCQ', n_sample_average=10):
        """ Analyse clock quality """
        key = 'clock'
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, key)
        if not trace:
            return
        clockQuality = trace.data
        clockQuality_warn_level = self.parameters.get('THRESHOLDS').get('clockquality_warn')
        clockQuality_fail_level = self.parameters.get('THRESHOLDS').get('clockquality_fail')

        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing Clock Quality check', flush=False)

        clockQuality_warn = np.where(clockQuality < clockQuality_warn_level)[0]
        clockQuality_fail = np.where(clockQuality < clockQuality_fail_level)[0]

        if len(clockQuality_warn) == 0 and len(clockQuality_fail) == 0:
            self.status_ok(key, detailed_message=f'ClockQuality={(clockQuality[-1])}')
            return

        last_val_average = np.nanmean(clockQuality[-n_sample_average:])

        # keep OK status if there are only minor warnings (lower warn level)
        warn_message = f'Trace {trace.get_id()}:'
        if len(clockQuality_warn) > 0:
            # try calculate number of warn peaks from gaps between indices
            n_qc_warn = self.calc_occurrences(clockQuality_warn)
            detailed_message = warn_message + f' {n_qc_warn}x Clock quality less then {clockQuality_warn_level}%' \
                               + self.get_last_occurrence_timestring(trace, clockQuality_warn)
            self.status_ok(key, detailed_message=detailed_message)

        # set WARN status for sever warnings in the past
        if len(clockQuality_fail) > 0:
            # try calculate number of fail peaks from gaps between indices
            n_qc_fail = self.calc_occurrences(clockQuality_fail)
            detailed_message = warn_message + f' {n_qc_fail}x Clock quality less then {clockQuality_fail_level}%' \
                               + self.get_last_occurrence_timestring(trace, clockQuality_fail)
            self.warn(key, detailed_message=detailed_message, count=n_qc_fail,
                      last_occurrence=self.get_last_occurrence(trace, clockQuality_fail))

        # set FAIL state if last value is less than fail level
        if last_val_average < clockQuality_fail_level:
            self.error(key, detailed_message=f'ClockQuality={(clockQuality[-1])}')

    def voltage_analysis(self, channel='VEI'):
        """ Analyse voltage channel for over/undervoltage """
        key = 'voltage'
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, key)
        if not trace:
            return
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
            detailed_message = warn_message + f' {n_undervolt}x Voltage under {low_volt}V ' \
                               + self.get_last_occurrence_timestring(trace, undervolt)
            self.warn(key, detailed_message=detailed_message, count=n_undervolt,
                      last_occurrence=self.get_last_occurrence(trace, undervolt))

    def pb_temp_analysis(self, channel='EX1'):
        """ Analyse PowBox temperature output. """
        key = 'temp'
        st = self.stream.select(channel=channel)
        trace = self.get_trace(st, key)
        if not trace:
            return
        voltage = trace.data * 1e-6
        thresholds = self.parameters.get('THRESHOLDS')
        temp = 20. * voltage - 20
        # average temp
        timespan = min([self.parameters.get('timespan') * 24 * 3600, int(len(temp) / trace.stats.sampling_rate)])
        nsamp_av = int(trace.stats.sampling_rate) * timespan
        av_temp_str = str(round(np.nanmean(temp[-nsamp_av:]), 1)) + deg_str
        # dt of average
        dt_t_str = str(timedelta(seconds=int(timespan))).replace(', 0:00:00', '')
        # current temp
        cur_temp = round(temp[-1], 1)
        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing PowBox temperature check (EX1)', flush=False)
            self.print(f'Average temperature at {np.nanmean(temp)}\N{DEGREE SIGN}', flush=False)
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

    def mass_analysis(self, channels=('VM1', 'VM2', 'VM3'), n_samp_mean=10):
        """ Analyse datalogger mass channels. """
        key = 'mass'

        # build stream with all channels
        st = Stream()
        for channel in channels:
            st += self.stream.select(channel=channel).copy()
        st.merge()

        # return if there are no three components
        if not len(st) == 3:
            return

        # correct for channel unit
        for trace in st:
            trace.data = trace.data * 1e-6  # TODO: Here and elsewhere: hardcoded, change this?

        # calculate average of absolute maximum of mass offset of last n_samp_mean
        last_values = np.array([trace.data[-n_samp_mean:] for trace in st])
        last_val_mean = np.nanmean(last_values, axis=1)
        common_highest_val = np.nanmax(abs(last_val_mean))
        common_highest_val = round(common_highest_val, 1)

        # get thresholds for WARN (max_vm_warn) and FAIL (max_vm_fail)
        thresholds = self.parameters.get('THRESHOLDS')
        max_vm_warn = thresholds.get('max_vm_warn')
        max_vm_fail = thresholds.get('max_vm_fail')
        if not max_vm_warn or not max_vm_fail:
            return

        # change status depending on common_highest_val
        if common_highest_val < max_vm_warn:
            self.status_ok(key, detailed_message=f'{common_highest_val}V')
        elif max_vm_warn <= common_highest_val < max_vm_fail:
            self.warn(key=key,
                      detailed_message=f'Warning raised for mass centering. Highest val (abs) {common_highest_val}V', )
        else:
            self.error(key=key,
                      detailed_message=f'Fail status for mass centering. Highest val (abs) {common_highest_val}V',)

        if self.verbosity > 1:
            self.print(40 * '-')
            self.print('Performing mass position check', flush=False)
            self.print(f'Average mass position at {common_highest_val}', flush=False)

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
                        n_occurrences = self.calc_occurrences(ind_array)
                        self.warn(key=key,
                                  detailed_message=f'Trace {trace.get_id()}: '
                                                   f'Found {n_occurrences} occurrence(s) of {volt_lvl}V: {key}: {message}'
                                                   + self.get_last_occurrence_timestring(trace, ind_array),
                                  count=n_occurrences,
                                  last_occurrence=self.get_last_occurrence(trace, ind_array))
                    # if last_val == current voltage (which is not 1) -> FAIL or last_val < 1: PBox no data
                    if volt_lvl == last_val or (volt_lvl == -1 and last_val < 1):
                        self.error(key, detailed_message=f'Last PowBox voltage state {last_val}V: {message}')

    def calc_occurrences(self, ind_array):
        # try calculate number of voltage peaks/plateaus from gaps between indices
        if len(ind_array) == 0:
            return 0
        else:
            # start index at 1 if there are gaps (n_peaks = n_gaps + 1)
            n_occurrences = 1

        min_samples = self.parameters.get('min_sample')
        if not min_samples:
            min_samples = 1

        # calculated differences in index array, diff > 1: gap, diff == 1: within peak/plateau
        diffs = np.diff(ind_array)
        gap_start_inds = np.where(np.diff(ind_array) > 1)[0]
        # iterate over all gaps and check "min_samples" before the gap
        for gsi in gap_start_inds:
            # right boundary index of peak (gap index - 1)
            peak_rb_ind = gsi - 1
            # left boundary index of peak
            peak_lb_ind = max([0, peak_rb_ind - min_samples])
            if all(diffs[peak_lb_ind: peak_rb_ind] == 1):
                n_occurrences += 1

        return n_occurrences

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
    def __init__(self, message=None, detailed_messages=None, count: int = 0, last_occurrence=None, show_count=True):
        if message is None:
            message = '-'
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
        self.is_active = False

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
