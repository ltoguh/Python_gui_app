#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:42:17 2026

@author: hugol
"""

import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from matplotlib.colors import hsv_to_rgb

# Import de tes fonctions utilitaires
from show_functions import show_retard, show_azimut, show_hsv, norm_01
from plugins import statistique, gaussien, median

class BirefringenceWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 1. Attributs de données (Accessibles partout via self) ---
        self.retard_data = None
        self.azimuth_data = None
        self.rgb_data = None
        self.current_data = None
        
        # --- 2. Configuration de l'interface ---
        self.setWindowTitle("OPTIMAG - Analyse de Biréfringence")
        self.resize(1200, 700)
        self.move(400, 200)
        self.init_ui()
        
        # --- 3. Lancement du calcul ---
        self.run_analysis()

    def init_ui(self):
        """ Crée les widgets de la fenêtre """
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Label d'information
        self.info_label = QtWidgets.QLabel("Initialisation...")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.info_label)

        # Barre de boutons
        self.btn_bar = QtWidgets.QHBoxLayout()
        self.menu = QtWidgets.QMenu()
        self.layout.addLayout(self.btn_bar)

        # On crée les boutons et on les connecte aux méthodes
        self.add_nav_button("Retard", self.display_retard)
        self.add_nav_button("Azimut", self.display_azimut)
        self.add_nav_button("HSV", self.display_hsv)
        #self.add_nav_button("Statistique", self.stat)
        self.add_nav_menu()

        # Viewer d'image Pyqtgraph
        self.im_view = pg.ImageView()
        self.im_view.setColorMap(pg.colormap.get('hot', source='matplotlib'))
        self.layout.addWidget(self.im_view)

    def add_nav_button(self, label, callback):
        btn = QtWidgets.QPushButton(label)
        btn.clicked.connect(callback)
        self.btn_bar.addWidget(btn)
        
    def add_nav_menu(self):        
        menu_button = QtWidgets.QPushButton("Plugins")
        menu_button.setMenu(self.menu)
        self.menu.addAction("Filtre Gaussien").triggered.connect(lambda: gaussien(self, self.im_view, self.info_label))
        self.menu.addAction("Filtre Médian").triggered.connect(lambda: median(self, self.im_view, self.info_label))
        self.menu.addAction("Statistique").triggered.connect(self.stat)
        self.btn_bar.addWidget(menu_button)

    def load_img(self, name):
        """ Charge et normalise une image """
        img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire {name}")
        denom = 65535.0 if img.dtype == np.uint16 else 255.0
        return img.astype(np.float64) / denom

    def run_analysis(self):
        
        try:
            S1, S2, S3, S4 = [], [], [], []
            bg1, bg2, bg3, bg4 = [], [], [], []
            Num = 1

            # Chargement des séquences
            for k in range(1, Num + 1):
                S1.append(self.load_img(f'img/25_fev_2025/b ({3+3*k}).tiff'))
                S2.append(self.load_img(f'img/25_fev_2025/b ({2+3*k}).tiff'))
                S3.append(self.load_img(f'img/25_fev_2025/b ({1+3*k}).tiff'))
                
                bg1.append(self.load_img(f'img/25_fev_2025/bg ({3+3*k}).tiff'))
                bg2.append(self.load_img(f'img/25_fev_2025/bg ({2+3*k}).tiff'))
                bg3.append(self.load_img(f'img/25_fev_2025/bg ({1+3*k}).tiff'))
                

            # Sélection de la zone (Crop)
            crop_img = cv2.imread('img/25_fev_2025/b (4).tiff', cv2.IMREAD_UNCHANGED)
            rect = cv2.selectROI("Select Crop Area", crop_img, fromCenter=False)
            cv2.destroyWindow("Select Crop Area")

            def apply_crop(lst, r):
                return [img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] for img in lst]

            S1, S2, S3 = [apply_crop(x, rect) for x in [S1, S2, S3]]
            bg1, bg2, bg3 = [apply_crop(x, rect) for x in [bg1, bg2, bg3]]

            # Calculs physiques
            beta = 10 * np.pi / 180
            gsumf, bfsum = 0, 0

            for k in range(Num):
                # Ton calcul spécifique (Sc1, Sc2...)
                Sc1, Sc2, Sc3 = S1[k], S2[k], S3[k]
                bgc1, bgc2, bgc3 = bg1[k], bg2[k], bg3[k]
                
                gck = ((Sc2-Sc1) + 1j*(2*Sc3-(Sc1+Sc2))) / (Sc1+Sc2)
                gbgk = ((bgc2-bgc1) + 1j*(2*bgc3-(bgc1+bgc2))) / (bgc1+bgc2)
                
                gsumf += gck - gbgk
                
                bfsum += 0.5 * (S1[k]+S2[k])

            # Stockage des résultats dans les attributs de la classe (self)
            gfmoy = gsumf / Num
            self.retard_data = np.tan(beta) * np.abs(gfmoy) * 530 / np.pi
            self.azimuth_data = (np.angle(gfmoy) / 2.0) * (180.0 / np.pi)

            # Calcul HSV
            H = (self.azimuth_data + 90) / 180
            S = np.ones_like(self.retard_data)
            V = norm_01(self.retard_data, 0, 3)
            hsv = np.stack((H, S, V), axis=-1)
            self.rgb_data = hsv_to_rgb(hsv)

            self.info_label.setText("Analyse terminée.")
            self.display_retard() # Affichage par défaut au lancement

        except Exception as e:
            self.info_label.setText(f"Erreur : {str(e)}")

    # --- Méthodes d'affichage (utilisent self.im_view) ---
    def display_retard(self):
        if self.retard_data is not None:
            show_retard(self.im_view, self.retard_data, self.info_label)

    def display_azimut(self):
        if self.azimuth_data is not None:
            show_azimut(self.im_view, self.azimuth_data, self.info_label)

    def display_hsv(self):
        if self.rgb_data is not None:
            show_hsv(self.im_view, self.rgb_data, self.info_label)
    
    def stat(self):
        statistique(self.im_view, self.info_label)

    def gaussien(self):
        gaussien(QtWidgets.QMainWindow,self.im_view, self.info_label)

    def median(self):
        median(QtWidgets.QMainWindow,self.im_view, self.info_label)

# Point d'entrée pour tester le module tout seul
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    gui = BirefringenceWindow()
    gui.show()
    sys.exit(app.exec())