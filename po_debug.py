import inspect
import os
import torch


torch.set_printoptions(threshold=500)

def debug_print(var, comment="", out=print):
    # if out == print:
    #     return None
        
    if out == print:
        out(comment, end='')
    else:
        out.write(comment)
    # Get the frame of the caller (the line that called debug_print)
    frame = inspect.currentframe().f_back

    # 1. Get the line number
    line_no = frame.f_lineno

    # 2. Extract variable name from the source code line
    # Note: inspect.stack()[1][4] returns the source code of the calling line
    line_code = inspect.stack()[1][4][0].strip()
    var_name = line_code.split('(')[1].split(')')[0]

    if out == print:
        out(f"line={line_no}, {var_name}={var}")
    else:
        out.write(f'line={line_no}, {var_name}={var}\n')
