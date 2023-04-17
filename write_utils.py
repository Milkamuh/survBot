from base64 import b64encode
from datetime import timedelta


def _convert_to_textstring(lst):
    return '\n'.join(lst)


def get_html_table_title(parameters):
    title = get_print_title_str(parameters)
    return f'<h3>{title}</h3>\n'


def get_html_text(text):
    return f'<p>{text}</p>\n'


def get_html_header(refresh_rate=10):
    header = ['<!DOCTYPE html>',
              '<html>',
              '<head>',
              '  <link rel="stylesheet" media="only screen and (max-width: 400px)" href="mobile.css" />',
              '  <link rel="stylesheet" media="only screen and (min-width: 401px)" href="desktop.css" />',
              '</head>',
              f'<meta http-equiv="refresh" content="{refresh_rate}" >',
              '<meta charset="utf-8">',
              '<meta name="viewport" content="width=device-width, initial-scale=1">',
              '<body>']
    header = _convert_to_textstring(header)
    return header


def get_mail_html_header():
    header = ['<html>',
              '<head>',
              '</head>',
              '<body>']
    header = _convert_to_textstring(header)
    return header


def init_html_table():
    return '<table style="width:100%">\n'


def finish_html_table():
    return '</table>\n'


def html_footer():
    footer = ['</body>',
              '</html>\n']
    footer = _convert_to_textstring(footer)
    return footer


def add_html_image(img_data, img_format='png'):
    return f"""<br>\n<img width="100%" src="data:image/{img_format};base64, {b64encode(img_data).decode('ascii')}">"""


def get_html_link(text, link):
    return f'<a href="{link}"> {text} </a>'


def get_html_row(items, html_key='td'):
    row_string = ''
    default_space = '  '
    row_string += default_space + '<tr>\n'
    for item in items:
        text = item.get('text')
        if item.get('bold'):
            text = '<b>' + text + '</b>'
        if item.get('italic'):
            text = '<i>' + text + '</i>'
        tooltip = item.get('tooltip')
        color = item.get('color')
        # check for black background of headers (shouldnt happen anymore)
        color = '#e6e6e6' if color == '#000000' else color
        hyperlink = item.get('hyperlink')
        text_str = get_html_link(text, hyperlink) if hyperlink else text
        html_class = item.get('html_class')
        class_str = f' class="{html_class}"' if html_class else ''
        row_string += 2 * default_space + f'<{html_key}{class_str} bgcolor="{color}" title="{tooltip}"> {text_str}'\
                    + f'</{html_key}>\n'
    row_string += default_space + '</tr>\n'
    return row_string


def get_print_title_str(parameters):
    timespan = parameters.get('timespan') * 24 * 3600
    tdelta_str = str(timedelta(seconds=int(timespan))).replace(', 0:00:00', '')
    return f'Analysis table of router quality within the last {tdelta_str}'
