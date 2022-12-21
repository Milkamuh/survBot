#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI overlay for the main survBot to show quality control of different stations specified in parameters.yaml file.
"""

__version__ = '0.1'
__author__ = 'Marcel Paffrath'

import os
import sys
import traceback

try:
    from PySide2 import QtGui, QtCore, QtWidgets
except ImportError:
    try:
        from PySide6 import QtGui, QtCore, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtGui, QtCore, QtWidgets
        except ImportError:
            raise ImportError('Could import neither of PySide2, PySide6 or PyQt5')

from matplotlib.figure import Figure

if QtGui.__package__ in ['PySide2', 'PyQt5', 'PySide6']:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
else:
    raise Exception('Not implemented')

from obspy import UTCDateTime

from survBot import SurveillanceBot
from write_utils import *
from utils import get_bg_color, modify_stream_for_plot, trace_yticks, trace_thresholds

try:
    from rest_api.utils import get_station_iccid
    from rest_api.rest_api_utils import get_last_messages, send_message, get_default_params
    sms_funcs = True
except ImportError:
    print('Could not load rest_api utils, SMS functionality disabled.')
    sms_funcs = False

deg_str = '\N{DEGREE SIGN}C'


class Thread(QtCore.QThread):
    """
    A simple thread that runs outside of the main event loop. Executes the function "runnable" and prevents
    freezing of the GUI. Run method is executed outside main event loop when called with thread.start().
    """
    update = QtCore.Signal()

    def __init__(self, parent, runnable, verbosity=0):
        super(Thread, self).__init__(parent=parent)
        self.setParent(parent)
        self.verbosity = verbosity
        self.runnable = runnable
        self.is_active = True

    def run(self):
        """ Try to run self.runnable and emit update signal, or print Exception if failed. """
        try:
            t0 = UTCDateTime()
            self.runnable()
            self.update.emit()
        except Exception as e:
            self.is_active = False
            print(e)
            print(traceback.format_exc())
        finally:
            if self.verbosity > 0:
                print(f'Time for Thread execution: {UTCDateTime() - t0}')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parameters='parameters.yaml'):
        """
        Main window of survBot GUI.
        :param parameters: Parameters dictionary file (yaml format)
        """
        super(MainWindow, self).__init__()

        # init some attributes
        self.last_mouse_loc = None
        self.status_message = ''
        self.starttime = UTCDateTime()

        # setup main layout of the GUI
        self.main_layout = QtWidgets.QVBoxLayout()
        self.centralWidget = QtWidgets.QWidget()
        self.centralWidget.setLayout(self.main_layout)
        self.setCentralWidget(self.centralWidget)

        # init new survBot instance, set parameters and refresh
        self.survBot = SurveillanceBot(parameter_path=parameters)
        self.parameters = self.survBot.parameters
        self.refresh_period = self.parameters.get('interval')
        self.dt_thresh = [int(val) for val in self.parameters.get('dt_thresh')]

        # create thread that is used to update
        self.thread = Thread(parent=self, runnable=self.survBot.execute_qc)
        self.thread.update.connect(self.fill_table)

        self.init_table()
        self.init_buttons()

        # These filters were used to track current mouse position if an event (i.e. mouseclick) is triggered
        self.table.installEventFilter(self)
        self.installEventFilter(self)

        # initiate clear_on_refresh flag and set status bar text
        self.clear_on_refresh = False
        self.fill_status_bar()

        # start thread that executes qc at first initiation, then activate timer for further thread activation
        self.thread.start()
        self.run_refresh_timer()

    def init_table(self):
        self.table = QtWidgets.QTableWidget()
        keys = self.survBot.keys
        station_list = self.survBot.station_list

        self.table.setColumnCount(len(keys))
        self.table.setRowCount(len(station_list))
        self.table.setHorizontalHeaderLabels(keys)

        for index, nwst_id in enumerate(station_list):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(nwst_id.rstrip('.')))
            item.setData(QtCore.Qt.UserRole, nwst_id)
            self.table.setVerticalHeaderItem(index, item)

        self.main_layout.addWidget(self.table)

        self.table.itemDoubleClicked.connect(self.plot_stream)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        if sms_funcs:
            self.table.verticalHeader().sectionClicked.connect(self.sms_context_menu)

        self.set_stretch()

    def init_buttons(self):
        if self.parameters.get('track_changes'):
            button_text = 'Clear track and refresh'
        else:
            button_text = 'Refresh'
        self.clear_button = QtWidgets.QPushButton(button_text)
        self.clear_button.setToolTip('Reset track changes and refresh table')
        self.clear_button.clicked.connect(self.refresh)
        self.main_layout.addWidget(self.clear_button)

    def refresh(self):
        self.set_clear_on_refresh()
        self.run_refresh_timer()
        self.thread.start()

    def run_refresh_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.thread.start)
        self.timer.start(int(self.refresh_period * 1e3))

    def eventFilter(self, object, event):
        """
        An event filter that stores last mouse position if an event is raised by the table. All events are passed
        to the parent class of the Mainwindow afterwards.
        """
        if hasattr(event, 'pos'):
            self.last_mouse_loc = event.pos()
        return super(QtWidgets.QMainWindow, self).eventFilter(object, event)

    def sms_context_menu(self, row_ind):
        """ Open a context menu when left-clicking vertical header item """
        header_item = self.table.verticalHeaderItem(row_ind)
        if not header_item:
           return
        nwst_id = header_item.data(QtCore.Qt.UserRole)

        context_menu = QtWidgets.QMenu()
        read_sms = context_menu.addAction('Get last SMS')
        send_sms = context_menu.addAction('Send SMS')
        action = context_menu.exec_(self.mapToGlobal(self.last_mouse_loc))
        if action == read_sms:
           self.read_sms(nwst_id)
        elif action == send_sms:
           self.send_sms(nwst_id)

    def read_sms(self, nwst_id):
        """ Read recent SMS over rest_api using whereversim portal """
        station = nwst_id.split('.')[1]
        iccid = get_station_iccid(station)
        if not iccid:
            print('Could not find iccid for station', nwst_id)
            return
        sms_widget = ReadSMSWidget(parent=self, iccid=iccid)
        sms_widget.setWindowTitle(f'Recent SMS of station: {nwst_id}')
        if sms_widget.data:
            sms_widget.show()
        else:
            self.notification('No recent messages found.')

    def send_sms(self, nwst_id):
        """ Send SMS over rest_api using whereversim portal """
        station = nwst_id.split('.')[1]
        iccid = get_station_iccid(station)

        sms_widget = SendSMSWidget(parent=self, iccid=iccid)
        sms_widget.setWindowTitle(f'Send SMS to station: {nwst_id}')
        sms_widget.show()

    def set_clear_on_refresh(self):
        self.clear_on_refresh = True

    def fill_status_bar(self):
        """ Set status bar text """
        self.status_message = self.survBot.status_message
        status_bar = self.statusBar()
        status_bar.showMessage(self.status_message)

    def fill_table(self):
        """ Fills the table with most recent information. Executed after execute_qc thread is done or on refresh. """

        # fill status bar first with new time
        self.fill_status_bar()

        for col_ind, check_key in enumerate(self.survBot.keys):
            for row_ind, nwst_id in enumerate(self.survBot.station_list):
                status_dict = self.survBot.analysis_results.get(nwst_id)
                status = status_dict.get(check_key)
                message, detailed_message = status.get_status_str()

                dt_thresh = [timedelta(seconds=sec) for sec in self.dt_thresh]
                bg_color = get_bg_color(check_key, status, dt_thresh)
                if check_key == 'temp':
                    if not type(message) in [str]:
                        message = str(message) + deg_str

                # Continue if nothing changed
                text = str(message)
                cur_item = self.table.item(row_ind, col_ind)
                if cur_item and text == cur_item.text():
                    if not self.parameters.get('track_changes') or self.clear_on_refresh:
                        # set item to default color/font and continue
                        self.set_font(cur_item)
                        self.set_fg_color(cur_item)
                    continue

                # Create new data item
                item = QtWidgets.QTableWidgetItem()
                item.setText(str(message))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setData(QtCore.Qt.UserRole, (nwst_id, check_key))

                # if text changed (known from above) set highlight color/font else (new init) set to default
                cur_item = self.table.item(row_ind, col_ind)
                if cur_item and check_key != 'last active':
                    self.set_fg_color(item, (0, 0, 0, 255))
                    self.set_font_bold(item)
                else:
                    self.set_fg_color(item)
                    self.set_font(item)

                # set item tooltip
                if detailed_message:
                    item.setToolTip(str(detailed_message))

                # set bg color corresponding to current text (OK/WARN/ERROR etc.)
                self.set_bg_color(item, bg_color)

                # insert new item
                self.table.setItem(row_ind, col_ind, item)

        # table filling/refreshing done, set clear_on_refresh to False
        self.clear_on_refresh = False

    def set_font_bold(self, item):
        """ Set item font bold """
        f = item.font()
        f.setWeight(QtGui.QFont.Bold)
        item.setFont(f)

    def set_font(self, item):
        """ Set item font normal """
        f = item.font()
        f.setWeight(QtGui.QFont.Normal)
        item.setFont(f)

    def set_bg_color(self, item, color):
        """ Set background color of item, color is RGBA tuple """
        color = QtGui.QColor(*color)
        item.setBackground(color)

    def set_fg_color(self, item, color=(20, 20, 20, 255)):
        """ Set foreground (font) color of item, color is RGBA tuple """
        color = QtGui.QColor(*color)
        item.setForeground(color)

    def set_stretch(self):
        hheader = self.table.horizontalHeader()
        for index in range(hheader.count()):
            hheader.setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)
        vheader = self.table.verticalHeader()
        for index in range(vheader.count()):
            vheader.setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)

    def plot_stream(self, item):
        nwst_id, check = item.data(QtCore.Qt.UserRole)
        st = self.survBot.data.get(nwst_id)
        if st:
            self.plot_widget = PlotWidget(self)
            self.plot_widget.setWindowTitle(nwst_id)
            st = modify_stream_for_plot(st, parameters=self.parameters)
            st.plot(equal_scale=False, method='full', block=False, fig=self.plot_widget.canvas.fig)
            # trace_ylabels(fig=self.plot_widget.canvas.fig, parameters=self.parameters)
            trace_yticks(fig=self.plot_widget.canvas.fig, parameters=self.parameters)
            trace_thresholds(fig=self.plot_widget.canvas.fig, parameters=self.parameters)
            self.plot_widget.show()

    def notification(self, text):
        mbox = QtWidgets.QMessageBox()
        mbox.setWindowTitle('Notification')
        #mbox.setDetailedText()
        mbox.setText(text)
        mbox.exec_()

    def closeEvent(self, event):
        self.thread.exit()
        event.accept()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class ReadSMSWidget(QtWidgets.QDialog):
    def __init__(self, iccid, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.table = QtWidgets.QTableWidget()
        self.layout().addWidget(self.table)
        self.resize(1280, 400)

        self.iccid = iccid
        self.data = self.print_sms_table()
        self.set_stretch()

    def print_sms_table(self, n=5, ntextbreak=40):
        messages = []
        params = get_default_params(self.iccid)
        for message in get_last_messages(params, n, only_delivered=False):
            messages.append(message)
        if not messages:
            return
        # pull dates to front
        keys = ['dateSent', 'dateModified', 'dateReceived']
        for item in messages[0].keys():
            if not item in keys:
                keys.append(item)
        self.table.setRowCount(n)
        self.table.setColumnCount(len(keys))
        self.table.setHorizontalHeaderLabels(keys)
        for row_index, message in enumerate(messages):
            for col_index, key in enumerate(keys):
                text = message.get(key)
                if type(text) == str and len(text) > ntextbreak:
                    textlist = list(text)
                    for index in range(ntextbreak, len(text), ntextbreak):
                        textlist.insert(index, '\n')
                    text = ''.join(textlist)
                item = QtWidgets.QTableWidgetItem()
                item.setText(str(text))
                self.table.setItem(row_index, col_index, item)
        return True

    def set_stretch(self):
        hheader = self.table.horizontalHeader()
        nheader = hheader.count()
        for index in range(nheader):
            if index < nheader - 1:
                hheader.setSectionResizeMode(index, QtWidgets.QHeaderView.ResizeToContents)
            else:
                hheader.setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)
        vheader = self.table.verticalHeader()
        for index in range(vheader.count()):
            vheader.setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)



class SendSMSWidget(QtWidgets.QDialog):
    def __init__(self, iccid, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.resize(400, 100)

        self.line_edit = QtWidgets.QLineEdit()
        self.main_layout.addWidget(self.line_edit)

        self.iccid = iccid

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                                    QtWidgets.QDialogButtonBox.Close)
        self.main_layout.addWidget(self.buttonBox)
        self.buttonBox.accepted.connect(self.send_sms)
        self.buttonBox.rejected.connect(self.reject)

    def send_sms(self):
        text = self.line_edit.text()
        params = get_default_params(self.iccid)
        send_message(params, text)
        self.close()


if __name__ == '__main__':
    program_path = sys.path[0]
    parameters = os.path.join(program_path, 'parameters.yaml')
    app = QtWidgets.QApplication([])
    window = MainWindow(parameters=parameters)
    window.showMaximized()
    sys.exit(app.exec_())
