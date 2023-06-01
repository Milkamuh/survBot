#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np

from obspy import Stream


def get_bg_color(check_key, status, dt_thresh=None, hex=False):
    message = status.message
    if check_key == 'last active':
        bg_color = get_time_delay_color(message, dt_thresh)
    elif check_key == 'temp':
        bg_color = get_temp_color(message)
    elif check_key == 'mass':
        bg_color = get_mass_color(message)
    else:
        if status.is_warn:
            bg_color = get_warn_color(status.count)
        elif status.is_error:
            if status.connection_error:
                bg_color = get_color('disc')
            else:
                bg_color = get_color('FAIL')
        else:
            bg_color = get_color(message)
    if not bg_color:
        bg_color = get_color('undefined')

    if hex:
        bg_color = '#{:02x}{:02x}{:02x}'.format(*bg_color[:3])
    return bg_color


def get_color(key):
    # some old GUI default colors
    # colors_dict = {'FAIL': (255, 85, 50, 255),
    #                'NO DATA': (255, 255, 125, 255),
    #                'WARN': (255, 255, 80, 255),
    #                'OK': (173, 255, 133, 255),
    #                'undefined': (230, 230, 230, 255),
    #                'disc': (255, 160, 40, 255),}
    colors_dict = {'FAIL': (195, 29, 14, 255),
                   'NO DATA': (255, 255, 125, 255),
                   'WARN': (250, 192, 63, 255),
                   'OK': (185, 245, 145, 255),
                   'undefined': (240, 240, 240, 255),
                   'disc': (126, 127, 131, 255), }
    return colors_dict.get(key)


def get_color_mpl(key):
    color_tup = get_color(key)
    return np.array([color/255. for color in color_tup])


def get_time_delay_color(dt, dt_thresh):
    """ Set color of time delay after thresholds specified in self.dt_thresh """
    if isinstance(dt, type(dt_thresh[0])):
        if dt < dt_thresh[0]:
            return get_color('OK')
        elif dt_thresh[0] <= dt < dt_thresh[1]:
            return get_color('WARN')
    return get_color('FAIL')


def get_warn_color(count, n_colors=20):
    if count >= n_colors:
        count = -1
    gradient = np.linspace((240, 245, 110, 255), (250, 192, 63, 255), n_colors, dtype=int)
    return tuple(gradient[count])


def get_mass_color(message):
    # can change this to something else if wanted. This way it always returns get_color (without warn count)
    if isinstance(message, (float, int)):
        return get_color('OK')
    return get_color(message)


def get_temp_color(temp, vmin=-10, vmax=60, cmap='coolwarm'):
    """ Get an rgba temperature value back from specified cmap, linearly interpolated between vmin and vmax. """
    if type(temp) in [str]:
        return get_color('undefined')
    cmap = matplotlib.cm.get_cmap(cmap)
    val = (temp - vmin) / (vmax - vmin)
    rgba = [int(255 * c) for c in cmap(val)]
    return rgba


def get_font_color(bg_color, hex=False):
    if hex:
        bg_color = matplotlib.colors.to_rgb(bg_color)
    bg_color_hsv = matplotlib.colors.rgb_to_hsv(bg_color)
    bg_color_hsl = hsv_to_hsl(bg_color_hsv)
    font_color = (255, 255, 255, 255) if bg_color_hsl[2] < 0.6 else (0, 0, 0, 255)
    if hex:
        font_color = '#{:02x}{:02x}{:02x}'.format(*font_color[:3])
    return font_color


def hsv_to_hsl(hsv):
    hue, saturation, value = hsv
    lightness = value * (1 - saturation / 2)
    saturation = 0 if lightness in (0, 1) else (value - lightness) / min(lightness, 1 - lightness)
    return hue, saturation, lightness


def modify_stream_for_plot(input_stream, parameters):
    """ copy (if necessary) and modify stream for plotting """

    # make a copy
    st = Stream()

    channels_dict = parameters.get('CHANNELS')

    # iterate over all channels and put them to new stream in order
    for index, ch_tup in enumerate(channels_dict.items()):
        # unpack tuple from items
        channel, channel_dict = ch_tup

        # get correct channel from stream
        st_sel = input_stream.select(channel=channel)
        # in case there are != 1 there is ambiguity
        if not len(st_sel) == 1:
            continue

        # make a copy to not modify original stream!
        tr = st_sel[0].copy()

        # multiply with conversion factor for unit
        unit_factor = channel_dict.get('unit')
        if unit_factor:
            tr.data = tr.data * float(unit_factor)

        # apply transformations if provided
        transform = channel_dict.get('transform')
        if transform:
            tr.data = transform_trace(tr.data, transform)

        # modify trace id to maintain plotting order
        name = channel_dict.get('name')
        tr.id = f'{index + 1}: {name} - {tr.id}'

        st.append(tr)

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


def set_axis_ylabels(fig, parameters, verbosity=0):
    """
    Adds channel names to y-axis if defined in parameters.
    """
    names = [channel.get('name') for channel in parameters.get('CHANNELS').values()]
    if not names: # or not len(st.traces):
        return
    if not len(names) == len(fig.axes):
        if verbosity:
            print('Mismatch in axis and label lengths. Not adding plot labels')
        return
    for channel_name, ax in zip(names, fig.axes):
        if channel_name:
            ax.set_ylabel(channel_name)


def set_axis_color(fig, color='0.8'):
    """
    Set all axes of figure to specific color
    """
    for ax in fig.axes:
        for key in ['bottom', 'top', 'right', 'left']:
            ax.spines[key].set_color(color)


def set_axis_yticks(fig, parameters, verbosity=0):
    """
    Adds channel names to y-axis if defined in parameters.
    """
    ticks = [channel.get('ticks') for channel in parameters.get('CHANNELS').values()]
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

        yticks = list(np.arange(ymin, ymax + step, step))
        ax.set_yticks(yticks)
        ax.set_ylim(ymin - 0.33 * step, ymax + 0.33 * step)


def plot_axis_thresholds(fig, parameters, verbosity=0):
    """
    Adds channel thresholds (warn, fail) to y-axis if defined in parameters.
    """
    if verbosity > 0:
        print('Plotting trace thresholds')

    keys_colors = {'warn': dict(color=0.8 * get_color_mpl('WARN'), linestyle=(0, (5, 10)), alpha=0.5, linewidth=0.7),
                   'fail': dict(color=0.8 * get_color_mpl('FAIL'), linestyle='solid', alpha=0.5, linewidth=0.7)}

    for key, kwargs in keys_colors.items():
        channel_threshold_list = [channel.get(key) for channel in parameters.get('CHANNELS').values()]
        if not channel_threshold_list:
            continue
        plot_threshold_lines(fig, channel_threshold_list, parameters, **kwargs)


def plot_threshold_lines(fig, channel_threshold_list, parameters, **kwargs):
    for channel_thresholds, ax in zip(channel_threshold_list, fig.axes):
        if not channel_thresholds:
            continue

        if not isinstance(channel_thresholds, (list, tuple)):
            channel_thresholds = [channel_thresholds]

        for warn_thresh in channel_thresholds:
            if isinstance(warn_thresh, str):
                warn_thresh = parameters.get('THRESHOLDS').get(warn_thresh)
            if type(warn_thresh in (float, int)):
                ax.axhline(warn_thresh, **kwargs)
