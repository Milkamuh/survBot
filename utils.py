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

