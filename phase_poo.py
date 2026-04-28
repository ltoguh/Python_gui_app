#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:56:06 2026

@author: hugol
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:42:17 2026

@author: hugol
"""

import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from matplotlib.colors import hsv_to_rgb

# Import de tes fonctions utilitaires
from show_functions import show_phase, show_BF

class PhaseWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 1. Attributs de données (Accessibles partout via self) ---
        self.retard_data = None
        self.azimuth_data = None
        self.rgb_data = None
        self.current_data = None
        
        # --- 2. Configuration de l'interface ---
        self.setWindowTitle("OPTIMAG - Analyse de l'image de phase ")
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
        self.layout.addLayout(self.btn_bar)

        # On crée les boutons et on les connecte aux méthodes
        self.add_nav_button("Phase", self.display_phase)
        self.add_nav_button("Brightfield", self.display_BF)

        # Viewer d'image Pyqtgraph
        self.im_view = pg.ImageView()
        self.im_view.setColorMap(pg.colormap.get('gray', source='matplotlib'))
        self.layout.addWidget(self.im_view)

    def add_nav_button(self, label, callback):
        btn = QtWidgets.QPushButton(label)
        btn.clicked.connect(callback)
        self.btn_bar.addWidget(btn)

    def load_img(self, name):
        """ Charge et normalise une image """
        img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire {name}")
        denom = 65535.0 if img.dtype == np.uint16 else 255.0
        return img.astype(np.float32) / denom

    def run_analysis(self):
        """ Ta logique mathématique complète """
        try:
            S1, S2, S3, S4 = [], [], [], []
            bg1, bg2, bg3, bg4 = [], [], [], []
            Num = 1

            # Chargement des séquences
            for k in range(1, Num + 1):
                S1.append(self.load_img(f'img/b ({4+4*k}).tiff'))
                S2.append(self.load_img(f'img/b ({2+4*k}).tiff'))
                S3.append(self.load_img(f'img/b ({1+4*k}).tiff'))
                S4.append(self.load_img(f'img/b ({3+4*k}).tiff'))
                bg1.append(self.load_img(f'img/bg ({4+4*k}).tiff'))
                bg2.append(self.load_img(f'img/bg ({2+4*k}).tiff'))
                bg3.append(self.load_img(f'img/bg ({1+4*k}).tiff'))
                bg4.append(self.load_img(f'img/bg ({3+4*k}).tiff'))

            # Sélection de la zone (Crop)
            crop_img = cv2.imread('img/b (7).tiff', cv2.IMREAD_UNCHANGED)
            rect = cv2.selectROI("Select Crop Area", crop_img, fromCenter=False)
            cv2.destroyWindow("Select Crop Area")

            def apply_crop(lst, r):
                return [img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] for img in lst]

            S1, S2, S3, S4 = [apply_crop(x, rect) for x in [S1, S2, S3, S4]]
            bg1, bg2, bg3, bg4 = [apply_crop(x, rect) for x in [bg1, bg2, bg3, bg4]]

            # Calculs physiques
            beta = 10 * np.pi / 180
            gsumf, psum, bfsum = 0, 0, 0

            for k in range(Num):
                # Ton calcul spécifique (Sc1, Sc2...)
                Sc1, Sc2, Sc3 = S1[k], S2[k], S3[k]
                bgc1, bgc2, bgc3 = bg1[k], bg2[k], bg3[k]
                
                pk = (S4[k]-S1[k])   / (S1[k]+S2[k])
                pbgk = (bg4[k]-bg1[k]) / (bg1[k]+bg2[k])
                
                psum += pk - pbgk
                bfsum += 0.5 * (S1[k]+S2[k])

            # Stockage des résultats dans les attributs de la classe (self)
            self.phase_data = psum / Num
            self.bf_data = bfsum / Num

            self.info_label.setText("Analyse terminée.")
            self.display_BF() # Affichage par défaut au lancement

        except Exception as e:
            self.info_label.setText(f"Erreur : {str(e)}")

    # --- Méthodes d'affichage (utilisent self.im_view) ---
    def display_phase(self):
        if self.phase_data is not None:
            show_phase(self.im_view, self.phase_data, self.info_label)

    def display_BF(self):
        if self.bf_data is not None:
            show_BF(self.im_view, self.bf_data, self.info_label)

# Point d'entrée pour tester le module tout seul
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    gui = PhaseWindow()
    gui.show()
    sys.exit(app.exec())