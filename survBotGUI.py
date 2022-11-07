#! /usr/bin/env python
"""
GUI overlay for the main survBot to show quality control of different stations specified in parameters.yaml file.
"""

__version__ = '0.1'
__author__ = 'Marcel Paffrath'

import os
import sys
import traceback
from datetime import timedelta

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

import matplotlib
from matplotlib.figure import Figure

if QtGui.__package__ in ['PySide2', 'PyQt5', 'PySide6']:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
else:
    raise Exception('Not implemented')

from obspy import UTCDateTime

from survBot import SurveillanceBot

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
    def __init__(self, parameters='parameters.yaml', dt_thresh=(300, 1800)):
        """
        Main window of survBot GUI.
        :param parameters: Parameters dictionary file (yaml format)
        :param dt_thresh: threshold for timing delay colourisation (yellow/red)
        """
        super(MainWindow, self).__init__()

        # some GUI default colors
        self.colors_dict = {'FAIL': (255, 50, 0, 255),
                            'NO DATA': (255, 255, 125, 255),
                            'WARN': (255, 255, 125, 255),
                            'WARNX': lambda x: (min([255, 200 + x**2]), 255, 125, 255),
                            'OK': (125, 255, 125, 255),
                            'undefined': (230, 230, 230, 255)}

        # init some attributes
        self.dt_thresh = dt_thresh
        self.last_mouse_loc = None
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

        for index, st_id in enumerate(station_list):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(st_id.rstrip('.')))
            item.setData(QtCore.Qt.UserRole, st_id)
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
        st_id = header_item.data(QtCore.Qt.UserRole)

        context_menu = QtWidgets.QMenu()
        read_sms = context_menu.addAction('Get last SMS')
        send_sms = context_menu.addAction('Send SMS')
        action = context_menu.exec_(self.mapToGlobal(self.last_mouse_loc))
        if action == read_sms:
           self.read_sms(st_id)
        elif action == send_sms:
           self.send_sms(st_id)

    def read_sms(self, st_id):
        """ Read recent SMS over rest_api using whereversim portal """
        station = st_id.split('.')[1]
        iccid = get_station_iccid(station)
        if not iccid:
            print('Could not find iccid for station', st_id)
            return
        sms_widget = ReadSMSWidget(parent=self, iccid=iccid)
        sms_widget.setWindowTitle(f'Recent SMS of station: {st_id}')
        if sms_widget.data:
            sms_widget.show()
        else:
            self.notification('No recent messages found.')

    def send_sms(self, st_id):
        """ Send SMS over rest_api using whereversim portal """
        station = st_id.split('.')[1]
        iccid = get_station_iccid(station)

        sms_widget = SendSMSWidget(parent=self, iccid=iccid)
        sms_widget.setWindowTitle(f'Send SMS to station: {st_id}')
        sms_widget.show()

    def set_clear_on_refresh(self):
        self.clear_on_refresh = True

    def fill_status_bar(self):
        """ Set status bar text """
        status_bar = self.statusBar()
        timespan = timedelta(seconds=int(self.parameters.get('timespan') * 24 * 3600))
        status_bar.showMessage(f'Program starttime (UTC) {self.starttime.strftime("%Y-%m-%d %H:%M:%S")} | '
                               f'Refresh period: {self.refresh_period}s | '
                               f'Showing data of last {timespan}')

    def fill_table(self):
        """ Fills the table with most recent information. Executed after execute_qc thread is done or on refresh. """
        for col_ind, check_key in enumerate(self.survBot.keys):
            for row_ind, st_id in enumerate(self.survBot.station_list):
                status_dict, detailed_dict = self.survBot.analysis_results.get(st_id)
                status = status_dict.get(check_key)
                detailed_message = detailed_dict.get(check_key)
                if check_key == 'last active':
                    bg_color = self.get_time_delay_color(status)
                elif check_key == 'temp':
                    bg_color = self.get_temp_color(status)
                    if not type(status) in [str]:
                        status = str(status) + deg_str
                else:
                    statussplit = status.split(' ')
                    if len(statussplit) > 1 and statussplit[0] == 'WARN':
                        x = int(status.split(' ')[-1].lstrip('(').rstrip(')'))
                        bg_color = self.colors_dict.get('WARNX')(x)
                    else:
                        bg_color = self.colors_dict.get(status)
                if not bg_color:
                    bg_color = self.colors_dict.get('undefined')

                # Continue if nothing changed
                text = str(status)
                cur_item = self.table.item(row_ind, col_ind)
                if cur_item and text == cur_item.text():
                    if not self.parameters.get('track_changes') or self.clear_on_refresh:
                        # set item to default color/font and continue
                        self.set_font(cur_item)
                        self.set_fg_color(cur_item)
                    continue

                # Create new data item
                item = QtWidgets.QTableWidgetItem()
                item.setText(str(status))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setData(QtCore.Qt.UserRole, (st_id, check_key))

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

    def get_time_delay_color(self, dt):
        """ Set color of time delay after thresholds specified in self.dt_thresh """
        dt_thresh = [timedelta(seconds=sec) for sec in self.dt_thresh]
        if dt < dt_thresh[0]:
            return self.colors_dict.get('OK')
        elif dt_thresh[0] <= dt < dt_thresh[1]:
            return self.colors_dict.get('WARN')
        return self.colors_dict.get('FAIL')

    def get_temp_color(self, temp, vmin=-10, vmax=60, cmap='coolwarm'):
        """ Get an rgba temperature value back from specified cmap, linearly interpolated between vmin and vmax. """
        if type(temp) in [str]:
            return self.colors_dict.get('undefined')
        cmap = matplotlib.cm.get_cmap(cmap)
        val = (temp - vmin) / (vmax - vmin)
        rgba = [int(255 * c) for c in cmap(val)]
        return rgba

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
        st_id, check = item.data(QtCore.Qt.UserRole)
        st = self.survBot.data.get(st_id)
        if st:
            self.plot_widget = PlotWidget(self)
            self.plot_widget.setWindowTitle(st_id)
            st.plot(equal_scale=False, method='full', block=False, fig=self.plot_widget.canvas.fig)
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
