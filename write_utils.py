from datetime import timedelta

def write_html_table_title(fobj, parameters):
    title = get_print_title_str(parameters)
    fobj.write(f'<h3>{title}</h3>\n')

def write_html_text(fobj, text):
    fobj.write(f'<p>{text}</p>\n')

def write_html_header(fobj):
    header = ['<!DOCTYPE html>',
              '<html>',
              '<style>',
              'table, th, td {',
              'border:1px solid black;',
              '}',
              '</style>',
              '<body>']
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
    fobj.write('<tr>\n')
    for item in items:
        text = item.text()
        color = item.backgroundColor().name()
        # fix for black background of headers
        color = '#e6e6e6' if color == '#000000' else color
        fobj.write(f'<{html_key} bgcolor="{color}">' + text + f'</{html_key}>\n')
    fobj.write('</tr>\n')

def get_print_title_str(parameters):
    timespan = parameters.get('timespan') * 24 * 3600
    tdelta_str = str(timedelta(seconds=int(timespan)))
    return f'Analysis table of router quality within the last {tdelta_str}'

