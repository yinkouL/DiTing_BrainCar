import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psychopy import monitors
import numpy as np
from metabci.brainstim.framework import Experiment
from metabci.brainstim.paradigm  import SSVEP, paradigm
from psychopy.tools.monitorunittools import deg2pix
from math import sin, cos, pi


if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])#设置背景颜色
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    # Experiment 对象，并配置了监视器、背景颜色、屏幕 ID、窗口大小和其他设置。
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0, # 值 0 通常指主要或默认显示器
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=True,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,  # 实验将记录在实验过程中显示的帧
        disable_gc=False,
        process_priority="normal", # 设置了实验进程的优先级
        use_fbo=False,
    )

    win = ex.get_window() # 允许实验代码访问和操作与 Experiment 实例关联的窗口

    # press q to exit paradigm interface
    # n_elements, rows, columns = 40, 5, 8
    n_elements, rows, columns = 6, 2, 3 # 直升机所
    stim_length ,stim_width= 200, 200
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 60                                                 # screen refresh rate
    stim_time = 2                                             # stimulus duration修改刺激时长
    stim_opacities = 1                                          # stimulus contrast
    # freqs = np.arange(25, 37, 0.3)                               # Frequency of instruction
    # phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # Phase of the instruction
    freqs = np.arange(8, 14, 1) # 直升机所
    phases = [0, 0, 0, 0, 0, 0]  # 0°, 120°, 240° [0, 2 * np.pi / 3, 4 * np.pi / 3]

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
        stim_length=stim_length, stim_width=stim_width)

    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
        stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs, phases=phases)
    basic_ssvep.config_index()
    basic_ssvep.config_response()       
    basic_ssvep.config_response_index()
    bg_color = np.array([-1, -1, -1])                           # background color
    display_time = 2  # 刺激时间
    index_time = 0.5  # 提示时间
    rest_time = 0.1   # 休息时间
    response_time = 0.1
    # port_addr = "COM7"

    port_addr = None 			                                 # Collect host ports
    # nrep = 5 # 初始化为 1 意味着默认情况下只需要执行一次

    nrep = 20 # 直升机所
    # lsl_source_id = 'meta_online_worker666'
    # online = True
    lsl_source_id = None#实验室流媒体层（LSL）流的源ID，用于实时数据采集
    online = False
    ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color,
        display_time=display_time,  index_time=index_time, rest_time=rest_time, response_time=response_time,
        port_addr=port_addr, nrep=nrep,  pdim='ssvep', lsl_source_id=lsl_source_id, online=online,
        ds_flag=False, robot_flag=True)

    ex.run()
