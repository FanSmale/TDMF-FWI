# -*- coding: utf-8 -*-
"""
Created on 2023/10/20 9:05

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import matplotlib.pylab as plt
import matplotlib as mpl

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.use('TkAgg')

import numpy as np
import cv2


font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


def pain_seg_seismic_data(para_seismic_data):
    """
    Plotting seismic data images of SEG salt datasets

    :param para_seismic_data:  Seismic data (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8), dpi = 120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels = [0,0.75,1.5,2.25,3.0], size=21)
    ax.set_yticklabels(labels = [0.0,0.50,1.00,1.50,2.00], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


def pain_openfwi_seismic_data(para_seismic_data):
    """
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    """
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.1, 8), dpi = 120)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()
    plt.close()


def pain_openfwi_velocity_model(para_velocity_model):
    """
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)

    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)

    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.35)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
    plt.show()

def pain_openfwi_velocity_model1(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()
    #plt.savefig('DST-FWI_Res49.png')


def pain_marmousi_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 17, 3.5, 0], vmin=min_velocity, vmax=max_velocity, cmap='jet')

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xticks([17], minor=True)
    ax.set_yticks(np.linspace(0, 3.5, 8))
    ax.set_xticklabels(labels=[' ', 2, 4, 6, 8, 10, 12, 14, 16], size=12)
    ax.set_yticklabels(labels=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], size=12)
    x_start, x_end = 7350, 8050
    y_start, y_end = 420, 280
    rect = patches.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, linewidth=10, edgecolor='r', facecolor='none')

    # 将矩形补丁添加到坐标轴上
    ax.add_patch(rect)


    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()

def pain_openfwi_seismic_data1(para_seismic_data, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    '''

    # The size of 1000 x 70 is not easy to display, we compressed it to a similar size of 400 x 301 as the SEG dataset.
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)   #

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi = 120)
    else:
        fig, ax = plt.subplots(figsize=(6.2, 8), dpi = 120)

    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)
    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')

        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


def pain_seg_velocity_model(para_velocity_model):
    """
    :param para_velocity_model: Velocity model (200 x 301) (numpy)
    :param min_velocity: Upper limit of velocity in the velocity model
    :param max_velocity: Lower limit of velocity in the velocity model
    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.3), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 3, 2, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.12, top=0.95, left=0.11, right=0.99)

    plt.show()


def PlotComparison_openfwi_velocity_model(para_raw_data, para_genenrate_data):
    """
        Compare raw seismic data (ground truth) and genenrated ones (prediction) of openfwi vmodel.

        :param para_raw_data: ground truth (70 x 70) (numpy)
        :param para_genenrate_data: prediction vmodel (70 x 70) (numpy)

        :return:
    """
    vmin = np.min(para_raw_data)
    vmax = np.max(para_raw_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    im1 = ax1.imshow(para_raw_data, vmin=vmin, vmax=vmax, extent=[0, 3, 2, 0])

    ax1.set_xlabel('Position (km)', font18)
    ax1.set_ylabel('Depth (km)', font18)
    ax1.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im1, ax=ax1, cax=cax, orientation='horizontal')

    im2 = ax2.imshow(para_genenrate_data, vmin=vmin, vmax=vmax, extent=[0, 3, 2, 0])

    ax2.set_xlabel('Position (km)', font18)
    ax2.set_ylabel('Depth (km)', font18)
    ax2.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im2, ax=ax2, cax=cax, orientation='horizontal')

    plt.subplots_adjust(bottom=0.12, top=0.95, left=0.11, right=0.99)

    plt.show()


def PlotComparison_openfwi_seismic_data(para_raw_data, para_genenrate_data):
    """
        Compare raw seismic data images and genenrated ones of openfwi dataset

        :param para_raw_data: 15 Hz Velocity model (1000 x 70) (numpy)
        :param para_genenrate_data: different observation systems, different main frequencies Velocity model (1000 x 70) (numpy)

        :return:
    """

    data = cv2.resize(para_raw_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16), dpi=120)
    im = ax1.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax1.set_xlabel('Position (km)', font21)
    ax1.set_ylabel('Time (s)', font21)

    ax1.set_xticks(np.linspace(0, 0.7, 5))
    ax1.set_yticks(np.linspace(0, 1.0, 5))
    ax1.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax1.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax1, cax=cax1, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    generate_data = cv2.resize(para_genenrate_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    im = ax2.imshow(generate_data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax2.set_xlabel('Position (km)', font21)
    ax2.set_ylabel('Time (s)', font21)

    ax2.set_xticks(np.linspace(0, 0.7, 5))
    ax2.set_yticks(np.linspace(0, 1.0, 5))
    ax2.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax2.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax2, cax=cax2, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    fig.tight_layout()

    plt.show()



