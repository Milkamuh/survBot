from datetime import timedelta

def write_html_table_title(fobj, parameters):
    title = get_print_title_str(parameters)
    fobj.write(f'<h3>{title}</h3>\n')

def write_html_text(fobj, text):
    fobj.write(f'<p>{text}</p>\n')

def write_html_header(fobj, refresh_rate=10):
    header = ['<!DOCTYPE html>',
              '<html>',
              f'<meta http-equiv="refresh" content="{refresh_rate}" >',
              '<meta charset="utf-8">',
              '<body>']
    # style = ['<style>',
    #           'table, th, td {',
    #           'border:1px solid black;',
    #           '}',
    #           '</style>',]
    for item in header:
        fobj.write(item + '\n')

def init_html_table(fobj):
    fobj.write('<table style="width:100%">\n')

def finish_html_table(fobj):
    fobj.write('</table>\n')

def write_html_footer(fobj):
    footer = ['</body>',
              '</html>']
    for item in footer:
        fobj.write(item + '\n')

def write_html_row(fobj, items, html_key='td'):
    default_space = '  '
    fobj.write(default_space + '<tr>\n')
    for item in items:
        text = item.get('text')
        tooltip = item.get('tooltip')
        color = item.get('color')
        # check for black background of headers (shouldnt happen anymore)
        color = '#e6e6e6' if color == '#000000' else color
        hyperlink = item.get('hyperlink')
        image_str = f'<a href="{hyperlink}">' if hyperlink else ''
        fobj.write(2 * default_space + f'<{html_key} bgcolor="{color}" title="{tooltip}"> {image_str}'
                   + text + f'</{html_key}>\n')
    fobj.write(default_space + '</tr>\n')

def get_print_title_str(parameters):
    timespan = parameters.get('timespan') * 24 * 3600
    tdelta_str = str(timedelta(seconds=int(timespan)))
    return f'Analysis table of router quality within the last {tdelta_str}'
