#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:22:38 2026

@author: hugol
"""

import pyqtgraph as pg
import numpy as np

def show_BF(view, im, label):
    global current_data
    current_data = im

    view.setImage(current_data)
    view.setColorMap(pg.colormap.get('gray', source='matplotlib'))
    label.setText("Image brightfield")

def show_retard(view, im, label):
    global current_data
    current_data = im

    view.setImage(current_data)
    view.setColorMap(pg.colormap.get('hot', source='matplotlib'))
    label.setText("Image de retard")
    
def show_azimut(view, im, label):
    global current_data
    current_data = im
        
    view.setImage(current_data)
    view.setColorMap(pg.colormap.get('hsv', source='matplotlib'))
    label.setText("Image de l'azimut")
    
def show_phase(view, im, label):
    global current_data
    current_data = im
    view.setImage(current_data)
    view.setColorMap(pg.colormap.get('gray', source='matplotlib'))
    label.setText("Image du gradient de phase")
     
def show_hsv(view, im, label):
    hsv_img = (im*255).astype(np.uint8)
    view.setImage(hsv_img)
    label.setText("Image HSV")
    
def norm_01(data, vmin=None, vmax=None):
    # Si on ne donne pas de bornes, on utilise les percentiles (auto)
    if vmin is None: vmin = np.percentile(data, 1)
    if vmax is None: vmax = np.percentile(data, 99)
    
    # On applique la normalisation sur la plage choisie
    return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)

