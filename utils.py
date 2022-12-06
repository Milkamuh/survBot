#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib


def get_bg_color(check_key, status, dt_thresh=None, hex=False):
    message = status.message
    if check_key == 'last active':
        bg_color = get_time_delay_color(message, dt_thresh)
    elif check_key == 'temp':
        bg_color = get_temp_color(message)
    else:
        if status.is_warn:
            bg_color = get_color('WARNX')(status.count)
        elif status.is_error:
            bg_color = get_color('FAIL')
        else:
            bg_color = get_color(message)
    if not bg_color:
        bg_color = get_color('undefined')

    if hex:
        bg_color = '#{:02x}{:02x}{:02x}'.format(*bg_color[:3])
    return bg_color


def get_color(key):
    # some GUI default colors
    colors_dict = {'FAIL': (255, 50, 0, 255),
                   'NO DATA': (255, 255, 125, 255),
                   'WARN': (255, 255, 80, 255),
                   'WARNX': lambda x: (min([255, 200 + x ** 2]), 255, 80, 255),
                   'OK': (125, 255, 125, 255),
                   'undefined': (230, 230, 230, 255)}
    return colors_dict.get(key)


def get_time_delay_color(dt, dt_thresh):
    """ Set color of time delay after thresholds specified in self.dt_thresh """
    if dt < dt_thresh[0]:
        return get_color('OK')
    elif dt_thresh[0] <= dt < dt_thresh[1]:
        return get_color('WARN')
    return get_color('FAIL')


def get_temp_color(temp, vmin=-10, vmax=60, cmap='coolwarm'):
    """ Get an rgba temperature value back from specified cmap, linearly interpolated between vmin and vmax. """
    if type(temp) in [str]:
        return get_color('undefined')
    cmap = matplotlib.cm.get_cmap(cmap)
    val = (temp - vmin) / (vmax - vmin)
    rgba = [int(255 * c) for c in cmap(val)]
    return rgba


def modify_stream_for_plot(st, parameters):
    """ copy (if necessary) and modify stream for plotting """
    ch_units = parameters.get('CHANNEL_UNITS')
    ch_transf = parameters.get('CHANNEL_TRANSFORM')

    # if either of both are defined make copy
    if ch_units or ch_transf:
        st = st.copy()

    # modify trace for plotting by multiplying unit factor (e.g. 1e-3 mV to V)
    if ch_units:
        for tr in st:
            channel = tr.stats.channel
            unit_factor = ch_units.get(channel)
            if unit_factor:
                tr.data = tr.data * float(unit_factor)
    # modify trace for plotting by other arithmetic expressions
    if ch_transf:
        for tr in st:
            channel = tr.stats.channel
            transf = ch_transf.get(channel)
            if transf:
                tr.data = transform_trace(tr.data, transf)

    return st


def transform_trace(data, transf):
    """
    Transform trace with arithmetic operations in order, specified in transf
    @param data: numpy array
    @param transf: list of lists with arithmetic operations (e.g. [['*', '20'], ] -> multiply data by 20
    """
    # This looks a little bit hardcoded, however it is safer than using e.g. "eval"
    for operator_str, val in transf:
        if operator_str == '+':
            data = data + val
        elif operator_str == '-':
            data = data - val
        elif operator_str == '*':
            data = data * val
        elif operator_str == '/':
            data = data / val
        else:
            raise IOError(f'Unknown arithmethic operator string: {operator_str}')

    return data


def trace_ylabels(fig, parameters, verbosity=0):
    """
    Adds channel names to y-axis if defined in parameters.
    Can get mixed up if channel order in stream and channel names defined in parameters.yaml differ, but it is
    difficult to assess the correct order from Obspy plotting routing.
    """
    names = parameters.get('channel_names')
    if not names: # or not len(st.traces):
        return
    if not len(names) == len(fig.axes):
        if verbosity:
            print('Mismatch in axis and label lengths. Not adding plot labels')
        return
    for channel_name, ax in zip(names, fig.axes):
        if channel_name:
            ax.set_ylabel(channel_name)


def trace_yticks(fig, parameters, verbosity=0):
    """
    Adds channel names to y-axis if defined in parameters.
    Can get mixed up if channel order in stream and channel names defined in parameters.yaml differ, but it is
    difficult to assess the correct order from Obspy plotting routing.
    """
    ticks = parameters.get('CHANNEL_TICKS')
    if not ticks:
        return
    if not len(ticks) == len(fig.axes):
        if verbosity:
            print('Mismatch in axis tick and label lengths. Not changing plot ticks.')
        return
    for ytick_tripple, ax in zip(ticks, fig.axes):
        if not ytick_tripple:
            continue
        ymin, ymax, step = ytick_tripple

        yticks = list(range(ymin, ymax + step, step))
        ax.set_yticks(yticks)
        ax.set_ylim(ymin - step, ymax + step)