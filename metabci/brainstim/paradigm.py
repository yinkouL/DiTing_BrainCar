# -*- coding: utf-8 -*-
import datetime
import math

# load in basic modules
import os
import os.path as op
import queue
import string
from datetime import time
import time

import numpy as np
from math import pi
from psychopy import data, visual, event
from psychopy.visual.circle import Circle
from pylsl import StreamInlet, resolve_byprop
from sympy import false

from .utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy

from robomaster import robot



# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):#stim_color 是一个用来指定每个频率的 RGB 颜色值的参数,可以用来灵活地控制最终显示的颜色。
    """
    具有不同频率和相位的多个刺激被呈现给受试者,并且需要以正弦波的方式调制刺激的颜色。

    Sinusoidal approximate sampling method.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        freqs: list of float
            Frequencies of each stimulus.
        phases: list of float
            Phases of each stimulus.
        srate: int or float
            Refresh rate of screen.
        frames: int
            Flashing frames.
        stim_color: list
            Color of stimu.

    Returns
    ----------
        color: ndarray
            shape(frames, len(fre), 3)

    """
    # 这句话创建了一个时间向量 time,其中包含了从 0 到整个时间序列持续时间之间的 frames 个等间隔的时间点。
    # 这个时间向量将用于后续的计算,例如计算每个时间点的正弦波形。(frames - 1) / srate: 这个表达式计算了总的时间长度。
    # frames 是总的帧数,srate 是屏幕的刷新率(每秒帧数)。因此,(frames - 1) / srate 就是整个时间序列的持续时间。
    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3)) #这个三维数组 color 可以用来存储每个时间帧和每个频率下的 RGB 颜色值
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(#创建一个 3xN 的数组
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        # 这在用户想要显示只有部分色彩通道的刺激物，或者想要创建灰度刺激物（将所有三个通道设置为相同值）
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".
        根据指定的"类型"参数确定每个偏移点的颜色

    author: Jieyu Wu

    Created on: 2022-12-14

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        stim_num: int
            Number of stimuli dots of each target.
        type: int
            avep code.

    Returns
    ----------
        point: ndarray
            (stim_num, 3)

    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type - 1] = [1, 1, 1]
    point = np.array(point)
    return point


def pix2height(win_size, pix_num):
    height_num = pix_num / win_size[1]
    return height_num


def height2pix(win_size, height_num):
    pix_num = height_num * win_size[1]
    return pix_num


def code_sequence_generate(basic_code, sequences):
    """Quickly generate coding sequences for sub-stimuli using basic endcoding units and encoding sequences.
        使用基本的终端编码单元和编码序列快速生成子刺激的编码序列
    author: Jieyu Wu

    Created on: 2023-09-18

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        basic_code: list
            Each basic encoding unit in the encoding sequence.
        sequences: list of array
            Encoding sequences for basic_code.

    Returns
    ----------
        code: ndarray
            coding sequences for sub-stimuli.

    """

    code = []
    for seq_i in range(len(sequences)):
        code_list = []
        seq_length = len(sequences[seq_i])
        for code_i in range(seq_length):
            code_list.append(basic_code[sequences[seq_i][code_i]])
        code.append(code_list)
    code = np.array(code)
    return code#每一行表示一个子刺激的编码序列


# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create the interface to the stimulus interface and initialize the window parameters.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        win:
            The window object.
        win_size: ndarray, shape(width, high)
            The size of the window in pixels.
        stim_length: int
            The length of the stimulus block in pixels.
        stim_width: int
            The width of the stimulus block in pixels.
        n_elements: int
            Number of stimulus blocks.
        stim_pos: ndarray, shape([x, y],...)
            Customize the position of the stimulus blocks with an array length
            that corresponds to the number of stimulus blocks.
        stim_sizes: ndarray, shape([length, width],...)
            The size of the stimulus block, the length of which corresponds to the number of stimulus blocks.
        symbols: str
            Stimulate the text of characters in the block.
        text_stimuli:
            Configuration information required for paradigm characters.
        rect_response:
            Configuration information required for the rectangular feedback box.
        res_text_pos: tuple, shape (x, y)
            The character position of the online response.
        symbol_height: int
            The height of the feedback character.
        symbol_text: str
            The character text of the online response.
        text_response:
            Configuration information for the feedback character.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Set the number, position, and size parameters of the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            n_elements: int
                Number of stimulus blocks, default is 40.
            rows: int
                Sets the number of stimulus block rows.
            columns: int
                Set the number of stimulus block columns.
           stim_pos: ndarray, shape(x,y)
                自定义刺激块的位置，如果为“无”，则将其排列成矩形阵列。
                Customize the position of the stimulus block, if None then it will be arranged in a rectangular array.
            stim_length: int
                Length of stimulus.
            stim_width: int
                Width of stimulus.

        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions

        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
        # if stim_pos is not None:# 原来是if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0])
            # note that the origin point of the coordinate axis should be the center of your screen
            # (so the upper left corner is in Quadrant 2nd), and the larger the coordinate value,
            # the farther the actual position is from the center
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be automatically converted
            # 这行代码的效果是将实验中所有刺激物的位置初始化为(0, 0), 即显示屏或坐标系的原点或左上角
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the center of each block
            first_pos = (
                np.array([self.win_size[0] / columns, self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception("Too much blocks or too big the stimulus region!")
            for i in range(columns):
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that
            # need to be set on the screen
            # 将计算出的刺激位置转换为实际的屏幕坐标
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width
        self.columns = columns
        self.rows = rows

    def config_text(
        self, unit="pix", symbols=None, symbol_height=0, tex_color=[1, 1, 1]
    ):
        """Sets the characters within the stimulus block.

        update log:
            2022-06-26 by Jianhang Wu

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            symbols: str
                Edit character text.
            symbol_height: int
                The height of the character in pixels.
            tex_color: list, shape(red, green, blue)
                Set the character color, the value is between -1.0 and 1.0.

        Raises
        ----------
            Exception: Insufficient characters

        """

        # check number of symbols
        if (symbols is not None) and (len(symbols) >= self.n_elements):
            self.symbols = symbols
        # 这一判断为将要输出的字符的列表，当达到最大刺激时长时不输出其他
        elif self.n_elements <= 40:
            self.symbols = "".join(["↺", "←", "↑",  "↓", "↻", "→"])
            # self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])#1234567890+-*/
        else:
            raise Exception("Please input correct symbol list!")

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 2
        self.text_stimuli = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=tex_color,
                    units=unit,
                    height=symbol_height,
                    bold=False,
                    name=symbol,
                )
            )
        self.text_stimuli_flash = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli_flash.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=[-1, -1, -1,],
                    units=unit,
                    height=symbol_height,
                    bold=False,
                    name=symbol,
                )
            )

    def config_response(
        self,
        symbol_text=":",#这是 symbol_text 参数的默认值，表示将显示为在线响应的文本。如果在调用函数时未提供此参数的值，它将默认为 "Speller: "
        symbol_height=0,#表示响应符号的高度（是打出文字的高度）
        symbol_color=(1, 1, 1),#如果在调用函数时未提供此参数的值，它将默认为 (1, 1, 1)，即白色
        bg_color=[-1, -1, -1],#如果在调用函数时未提供此参数的值，它将默认为 (-1, -1, -1)，即黑色
    ):
        """Sets the character of the online response.

        update log:
            2022-08-10 by Wei Zhao

            2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

        Parameters
        ----------
            symbol_text: str
                Online response string.
            symbol_height: int
                Height of response symbol.
            symbol_color: tuple, shape (red, green, blue)
                The character color of the online response, the value is between -1.0 and 1.0.
            bg_color: list
                Color of background symbol.

        Raises
        ----------
            Exception: Insufficient characters
        """
        # brige_length的计算方式是取窗口宽度的中心，加上刺激位置的x坐标，减去刺激长度的一半。这有效地计算了从窗口中心到刺激左边缘的桥的长度
        # brige_length=117.0 brige_width=60.0
        # helicopter
        brige_length = self.stim_length / 2
        brige_width = self.win_size[1] / 9
        # normal
        # brige_length = self.win_size[0] / 2 + self.stim_pos[0][0] - self.stim_length / 2
        # brige_width = self.win_size[1] / 2 - self.stim_pos[0][1] - self.stim_width / 2

        # 创建了一个跨越窗口宽度（不包括"桥"区域）、高度为"桥"区域三倍的矩形视觉元素。该矩形垂直居中于窗口中，填充颜色为背景色，边框颜色为白色。
        # 这个视觉元素可能用作实验或模拟中的响应区域或视觉提示
        self.rect_response = visual.Rect(#创建在线反馈框
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 3 / 3,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),#这将矩形垂直居中于窗口中
            fillColor=bg_color,
            lineColor=[1, 1, 1],
        )

        self.res_text_pos = (#定义了第一个文本元素的位置(-784.5, 510.0)
            -self.win_size[0] / 3, #-self.win_size[0] / 2 + brige_length
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_pos = (#定义了文本元素的重置位置
            -self.win_size[0] / 3, #-self.win_size[0] / 2 + brige_length
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_text = ""
        if symbol_height == 0:
            self.symbol_height = brige_width
        self.symbol_text = symbol_text
        self.text_response = visual.TextStim(
            win=self.win,
            text=symbol_text,
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=self.symbol_height,
            bold=True,
        )


# config visual stimuli


class VisualStim(KeyboardInterface):
    """Create various visual stimuli.

    The subclass VisualStim inherits from the parent class KeyboardInterface, duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        index_stimuli:
            Configuration information for the target prompt.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    def config_index(self, index_height=0, units="pix"):
        """Config index stimuli: downward triangle (Unicode: \u2BC6)

        Parameters
        ----------
            index_height: int
                The height of the cue symbol, which defaults to half the height of the stimulus block.

        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="\u25CF",
            font="Arial",
            color=[1.0, -1.0, -1.0],
            colorSpace="rgb",
            units=units,
            height=index_height,
            bold=True,
            autoLog=False,
        )

    def config_response_index(self, index_height=0, units="pix"):
        """Config response_index stimuli: downward triangle (Unicode: \u2BC6)
            用绿色箭头显示用户盯到的目标

        Parameters
        ----------
            index_height: int
                The height of the cue symbol, which defaults to half the height of the stimulus block.

        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.response_index_stimuli = visual.TextStim(
            win=self.win,
            text="\u25CF",
            font="Arial",
            color=[0.0, 1.0, 0.0],#绿色
            colorSpace="rgb",
            units=units,
            height=index_height,
            bold=True,
            autoLog=False,#设置自动记录日志为False。在PsychoPy中，当autoLog=True时，会自动记录与刺激对象相关的事件信息到日志文件中，如刺激的呈现时间、位置等。
        )


# standard SSVEP paradigm


class SemiCircle(Circle):
    """
    A SemiCircle class inherited from Circle.
    """

    def _calcVertices(self):
        # only draw half of a circle
        d = np.pi / self.edges
        self.vertices = np.asarray(
            [
                np.asarray((np.sin(e * d), np.cos(e * d))) * self.radius
                for e in range(int(round(self.edges) + 1))
            ]
        )


# standard SSVEP paradigm


class SSVEP(VisualStim):
    """Create SSVEP stimuli.

    The subclass SSVEP inherits from the parent class VisualStim, and duplicate properties are not listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Time of stimulus flash
        stim_color: list, shape(red, green, blue)
            The color of the stimulus block, taking values between -1.0 and 1.0.
        stim_opacities: float
            Opacity, default opaque.
        stim_frames: int
            The number of frames contained in a single-trial stimulus.
        stim_oris: ndarray
            刺激快的方向
            Orientation of the stimulus block.
        stim_sfs: ndarray
            刺激块的空间频率。
            Spatial frequency of the stimulus block.
        stim_contrs: ndarray
            Stimulus block contrast.
        freqs: list, shape(fre, …)
            Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
        phases: list, shape(phase, …)
             Stimulus block flicker phase, length consistent with the number of stimulus blocks.
        stim_colors: list, shape(red, green, blue)
            The color configuration required for the stimulus block flashing.
        flash_stimuli:
            刺激块闪烁所需的配置信息。
            The configuration information required for the flashing of the stimulus block.

    Tip
    ----
     .. code-block:: python
        :caption: An example of creating SSVEP stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import SSVEP,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 20, 4, 5
        stim_length, stim_width = 150, 150
        stim_color, tex_color = [1,1,1], [1,1,1]
        fps = 120                                                   # screen refresh rate
        stim_time = 2                                               # stimulus duration
        stim_opacities = 1                                          # stimulus contrast
        freqs = np.arange(8, 16, 0.4)                               # Frequency of instruction
        phases = np.array([i*0.35%2 for i in range(n_elements)])    # Phase of the instruction
        basic_ssvep = SSVEP(win=win)
        basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns,
            stim_length=stim_length, stim_width=stim_width)
        basic_ssvep.config_text(tex_color=tex_color)
        basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid',
            stim_color=stim_color, stim_opacities=stim_opacities, freqs=freqs, phases=phases)
        basic_ssvep.config_index()
        basic_ssvep.config_response()
        bg_color = np.array([-1, -1, -1])                           # background color
        display_time = 1
        index_time = 0.5
        rest_time = 0.5
        response_time = 1
        port_addr = None 			                                 # Collect host ports
        nrep = 1
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color,
            display_time=display_time,  index_time=index_time, rest_time=rest_time, response_time=response_time,
            port_addr=port_addr, nrep=nrep,  pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        """Item class from VisualStim.

        Args:

        """
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(
        self,
        refresh_rate,
        stim_time,
        stim_color,
        stimtype="sinusoid",
        stim_opacities=1,
        **kwargs
    ):
        """Config color of stimuli.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_time: float
                Time of each stimulus.
            stim_color: int
                The color of the stimulus block.
            stimtype: str
                Stimulation flicker mode, default to sine sampling flicker.正弦刺激
            stim_opacities: float
                Opacity, default to opaque.不透明度
            freqs: list, shape(fre, …)
                刺激块闪烁频率、长度与刺激块数量一致。
                Stimulus block flicker frequency, length consistent with the number of stimulus blocks.
            phases: list, shape(phase, …)
                Stimulus block flicker phase, length consistent with the number of stimulus blocks.

        Raises
        ----------
            Exception: Inconsistent frames and color matrices

        """

        # initialize extra inputs
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        # 这段代码的目的是自动检测窗口的实际帧率并相应地设置 self.refresh_rate 变量。这在帧率事先未知或可能变化的情况下很有用，例如在不同硬件或环境上运行时。
        # nIdentical 参数指定用于计算的相同帧的数量，nWarmUpFrames 参数指定用于预热计算的帧数
        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        self.stim_oris = np.zeros((self.n_elements,))  # orientation用于存储每个刺激块的方向
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency用于存储每个刺激块的空间频率
        self.stim_contrs = np.ones((self.n_elements,))  # contrast所有元素都初始化为1 用于存储每个刺激块的对比度

        # check extra inputs允许在创建对象时通过 kwargs 参数来覆盖这些属性的默认值
        if "stim_oris" in kwargs.keys():
            self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs.keys():
            self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs.keys():
            self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs.keys():
            self.freqs = kwargs["freqs"]
        if "phases" in kwargs.keys():
            self.phases = kwargs["phases"]

        # check consistency
        if stimtype == "sinusoid":
            self.stim_colors = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames,
                    stim_color=stim_color,
                )
                - 1
            )
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception("Please input correct num of stims!")

        incorrect_frame = self.stim_colors.shape[0] != self.stim_frames
        incorrect_number = self.stim_colors.shape[1] != self.n_elements
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")

        # add flashing targets onto interface，使用 PsychoPy 库在屏幕上添加闪烁的刺激目标
        self.flash_stimuli = []
        # 在 for 循环中,它会创建 self.stim_frames 个闪烁刺激目标,并将它们添加到 self.flash_stimuli 列表中
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,#将刺激目标显示在 self.win 窗口上
                    units="pix",#使用像素作为单位
                    nElements=self.n_elements,#设置刺激目标的数量
                    sizes=self.stim_sizes,#设置每个刺激目标的大小
                    xys=self.stim_pos,#设置每个刺激目标的位置
                    #self.stim_colors[sf, ...] 是一种 NumPy 数组的高级索引方式。
                    # 它会选择 self.stim_colors 数组中第 sf 行的所有列(用 ... 表示)。
                    # 即对于每个闪烁刺激目标,它的颜色会根据当前处理的帧序号 sf 而变化。每个刺激目标在不同的帧中会显示不同的颜色。
                    colors=self.stim_colors[sf, ...],#设置每个刺激目标的颜色,根据当前帧 sf 进行变化
                    opacities=self.stim_opacities,
                    oris=self.stim_oris,
                    sfs=self.stim_sfs,
                    contrs=self.stim_contrs,
                    elementTex=np.ones((64, 64)),# 创建了一个64x64的NumPy数组，填充值为1 每个刺激元素将具有均匀的实心纹理
                    elementMask=None,# 不会应用任何遮罩，整个纹理将可见
                    texRes=48,# 64x64的纹理将缩放到48x48像素渲染在屏幕上
                )
            )

flag = []


# basic experiment control

class StreamProcessor:
    """
    Dynamic Stop Process
    """
    def __init__(self, inlet):
        self.inlet = inlet
        self.active = False
        self.current_mark = 0

    def start(self):
        self.inlet.open_stream()
        self.active = True

    def get_marker(self):
        """返回 (是否收到有效标记, 标记值)"""
        if not self.active:
            return False, 0

        try:
            samples, _ = self.inlet.pull_chunk()
            if samples:
                print(samples)
                self.current_mark = samples[-1][0]
                if self.current_mark > 0:
                    return True, self.current_mark
            return False, self.current_mark
        except Exception as e:
            print(f"streamerror: {str(e)}")
            return False, 0


class Robomaster:
    def __init__(self):
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")

        self.ep_chassis = ep_robot.chassis

        self.x_val = 0.5
        self.y_val = 0.6
        self.z_val = 90

    def command(self, x_val=0, y_val=0, z_val=0, xy_speed=1, z_speed=45, predict_id=None):
        if predict_id is not None:
            if predict_id == 0:
                z_val = 45
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action
            elif predict_id == 1:
                y_val = -0.2
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action
            elif predict_id == 2:
                x_val = 0.2
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action
            elif predict_id == 3:
                x_val = -0.2
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action
            elif predict_id == 4:
                z_val = -45
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action
            elif predict_id == 5:
                y_val = 0.2
                action = self.ep_chassis.move(x=x_val, y=y_val, z=z_val, xy_speed=xy_speed, z_speed=z_speed)
                return action


def paradigm(
    VSObject,#代表实验视觉刺激的对象
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,#在线实验期间反馈显示的持续时间
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,#在线处理程序的通信ID
    online=None,
    device_type="NeuroScan",
    ds_flag = False,
    robot_flag = False,
):
    """
    The classical paradigm is implemented, the task flow is defined, the ' q '
    exit paradigm is clicked, and the start selection interface is returned.

    author: Wei Zhao

    Created on: 2022-07-30

    update log:

        2022-08-10 by Wei Zhao

        2022-08-03 by Shengfu Wen

        2022-12-05 by Jie Mei

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        VSObject:
            Examples of the three paradigms.
        win:
            window.
        bg_color: ndarray
            Background color.
        fps: int
            Display refresh rate.
        display_time: float
            Keyboard display time before 1st index.
        index_time: float
            Indicator display time.
        rest_time: float, optional
            SSVEP and P300 paradigm: the time interval between the target cue and the start of the stimulus.
            MI paradigm: the time interval between the end of stimulus presentation and the target cue.
        respond_time: float, optional
            Feedback time during online experiment.
        image_time: float, optional,
            MI paradigm: Image time.
        port_addr:
             Computer port , hexadecimal or decimal.
        nrep: int
            Num of blocks.
        pdim: str
            One of the three paradigms can be 'ssvep ', ' p300 ', ' mi ' and ' con-ssvep '.
        mi_flag: bool
            Flag of MI paradigm.
        lsl_source_id: str
            The id of communication with the online processing program needs to be consistent between the two parties.
        online: bool
            Flag of online experiment.
        device_type: str
            See support device list in brainstim README file

    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type)
        )
    port_frame = int(0.05 * fps)

    # 该代码的目的是为实验设置视觉刺激和响应处理,并在实验在线进行时建立与外部数据源（如 EEG 放大器）的连接。
    # 采取的具体操作取决于范式维度和是否有可用的有效数据源。
    inlet = False
    if online:
        if (
            pdim == "ssvep"
        ):
            if robot_flag is True:
                robomaster = Robomaster()  # 示例化robomaster

            VSObject.text_response.text = copy(VSObject.reset_res_text)
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)
            VSObject.symbol_text = copy(VSObject.reset_res_text)
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id这通过 source_id 属性解析所有流,超时时间为 5 秒
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data

    # # 设置标签串口和标签
    # trigger = Trigger('COM9')
    # count = 0

    if pdim == "ssvep":
        # config experiment settings
        # 设置了实验的条件,并创建了一个试次处理器对象,用于在实验过程中按照随机顺序呈现这些条件
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="sequential")#随机改random顺序为sequential

        # start routine
        # episode 1: display speller interface
        # 在每一帧中,会执行以下操作:
        # 如果 online 标志为 True,则绘制 VSObject.rect_response（矩形）和 VSObject.text_response（文本）这两个视觉刺激。
        # 遍历 VSObject.text_stimuli 列表,并绘制其中的所有文本刺激。
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)#如果有端口对象 port 存在,则将其设置为 0。

        # total_start_time = datetime.datetime.now()  # 记录开始时间
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position（初始化提示索引的起始位置）
            # 在刺激位置的基础上,向上偏移 VSObject.stim_width / 2 的距离,得到最终的索引刺激位置。
            # 将 VSObject.index_stimuli 对象的位置设置为上一步计算得到的位置。
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])#这一步为更新三角指示的位置
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting（在盯刺激快之前移动提示索引的位置）)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()#响应矩形
                        VSObject.text_response.draw()#响应文本
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()


            # 创建队列并打开inlet
            if online:
                stream_processor = StreamProcessor(inlet)
                stream_processor.start()

            # total_time = 0
            # phase III: target stimulating
            # 在一定时间内(由 VSObject.stim_frames 决定)持续显示目标刺激,
            # 并在特定的帧数(由 port_frame 决定)向外部设备(由 port 表示)发送触发信号
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)

                # start_time = time.time()  # 记录开始时间
                # final
                if online and ds_flag:
                    received, mark = stream_processor.get_marker()
                    if received:
                        break

                VSObject.flash_stimuli[sf].draw()
                for text_stimulus_flash in VSObject.text_stimuli_flash:
                    text_stimulus_flash.draw()
                win.flip()
                # end_time = time.time()  # 记录结束时间
                # total_time = total_time + (end_time-start_time)
            # print("total_time for one trial:", total_time)

            # phase IV: respond
            # 如果inlet对象可用,接收在线预测,更新VSObject.symbol_text和res_text_pos变量,并显示指定的response_time内的响应
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()

                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

                # Dynamic stop
                if ds_flag:
                    predict_id = mark - 1 if mark > 0 else -1
                    print(f"预测标签为{predict_id}")
                    # mark = 0
                else:
                    samples, timestamp = inlet.pull_sample()
                    predict_id = int(samples[0]) - 1  # online predict id
                    print(f"预测标签为{predict_id}")

                # # 如果达到最大时长，则直接输出当前刺激文本的id（当前看哪个字母就输出哪个字母）
                # if predict_id == -1:
                #     predict_id = id
                if predict_id != -1:
                    if len(VSObject.symbol_text) >= 20: #如果字超出框就清空
                        VSObject.symbol_text = ""
                    VSObject.symbol_text = (
                        VSObject.symbol_text + VSObject.symbols[predict_id]
                    )
                    print("+ symbol", VSObject.symbols[predict_id])
                res_text_pos = (
                    res_text_pos[0] , #+ VSObject.symbol_height / 3
                    res_text_pos[1],#更新 res_text_pos 变量,用于控制文本响应的位置
                )

                # initialise response_index position（初始化用户盯到的目标的位置）
                position = VSObject.stim_pos[predict_id] + np.array([0, -VSObject.stim_width/2])  #
                VSObject.response_index_stimuli.setPos(position)

                action = robomaster.command(predict_id=predict_id) #小车运动
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    if predict_id != -1:VSObject.response_index_stimuli.draw() #显示用户盯到的目标
                    iframe += 1
                    win.flip()
                action.wait_for_completed()
                # robomaster.close()
        # total_end_time = datetime.datetime.now()
        # total_time = total_start_time - total_end_time
        # print(total_time)
