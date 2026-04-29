#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:42:17 2026

@author: hugol
"""

import numpy as np
import cv2
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from matplotlib.colors import hsv_to_rgb
from plugins import statistique, gaussien, median, binning, vectorial, SliderBlend, SliderHsvPhase, ScaleBar

# Import de tes fonctions utilitaires
from show_functions import show_retard, show_azimut, show_hsv, norm_01, show_phase, show_BF

class MultimodalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Attributs de données (Accessibles partout via self)
        self.retard_data = None
        self.azimuth_data = None
        self.phase_data = None
        self.rgb_data = None
        self.current_data = None
        self.slider_widget = None
        self.vec = None
        
        # Configuration de l'interface
        self.setWindowTitle("OPTIMAG - Analyse de Biréfringence")
        self.setWindowIcon(QtGui.QIcon("prism.jpg"))
        self.resize(1200, 700)
        self.move(400, 200)
        self.init_ui()
        
        self.setStyleSheet(""" 
        QMainWindow { background-color: #1F1F1F;
                     }
        QPushButton { background-color: #424242;
                     color: #DEDEDE;
                     border:none;
                     border-radius: 4px;
                     font: 20px;
                     }
        QPushButton:hover { background-color: #535454; }
        QPushButton:pressed { background-color: #21AEFC; }

        QLabel { color: #DEDEDE;
                font: 16px;
                font-weight: bold; }
            """)
        
        # Lancement du calcul
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
        self.add_nav_button("Phase", self.display_phase)
        self.add_nav_button("Brightfield", self.display_BF)
        self.add_nav_button("Vectoriel", lambda: vectorial(self, self.central_widget, self.retard_data, self.azimuth_data, self.gamma, self.phase_data, self.im_view, self.info_label))
        self.add_nav_button("Overlay", self.tout)
        self.add_nav_button("Slider", self.rideau)
        self.add_nav_menu()

        # Viewer d'image Pyqtgraph
        self.im_view = pg.ImageView()
        self.im_view.setColorMap(pg.colormap.get('hot', source='matplotlib'))
        self.layout.addWidget(self.im_view)
        self.open_file()

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
        self.menu.addAction("Binning").triggered.connect(lambda: binning(self, self.im_view, self.info_label))
        #self.menu.addAction("Vectoriel").triggered.connect(lambda: vectorial(self, self.retard_data, self.azimuth_data, self.gamma, self.phase_data, self.im_view, self.info_label))
        self.btn_bar.addWidget(menu_button)

    def load_img(self, name):
        """ Charge et normalise une image """
        img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire {name}")
        denom = 65535.0 if img.dtype == np.uint16 else 255.0
        return img.astype(np.float32) / denom
    
    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "", "Images (*.png *.jpg *.tiff *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            self.im_view.setImage(img)
            self.im_view.show()
            self.folder_name = os.path.dirname(file_path) #on vq chercher le chemin qui précède le fichier
            self.file_name = os.path.basename(file_path)
            

    def run_analysis(self):
        """ Ta logique mathématique complète """
        try:
            S1, S2, S3, S4 = [], [], [], []
            bg1, bg2, bg3, bg4 = [], [], [], []
            Num = 1

            # Chargement des séquences
            for k in range(1, Num + 1):
                self.new_path1 = os.path.join(self.folder_name, f"b ({4+4*k}).tiff")
                self.new_path2 = os.path.join(self.folder_name, f"b ({2+4*k}).tiff")
                self.new_path3 = os.path.join(self.folder_name, f"b ({1+4*k}).tiff")
                self.new_path4 = os.path.join(self.folder_name, f"b ({3+4*k}).tiff")
                self.new_path_bg1 = os.path.join(self.folder_name, f"bg ({4+4*k}).tiff")
                self.new_path_bg2 = os.path.join(self.folder_name, f"bg ({2+4*k}).tiff")
                self.new_path_bg3 = os.path.join(self.folder_name, f"bg ({1+4*k}).tiff")
                self.new_path_bg4 = os.path.join(self.folder_name, f"bg ({3+4*k}).tiff")
                
                S1.append(self.load_img(self.new_path1))
                S2.append(self.load_img(self.new_path2))
                S3.append(self.load_img(self.new_path3))
                S4.append(self.load_img(self.new_path4))
                bg1.append(self.load_img(self.new_path_bg1))
                bg2.append(self.load_img(self.new_path_bg2))
                bg3.append(self.load_img(self.new_path_bg3))
                bg4.append(self.load_img(self.new_path_bg4))

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
                
                gck = ((Sc2-Sc1) + 1j*(2*Sc3-(Sc1+Sc2))) / (Sc1+Sc2)
                gbgk = ((bgc2-bgc1) + 1j*(2*bgc3-(bgc1+bgc2))) / (bgc1+bgc2)
                pk = (S4[k]-S1[k])   / (S1[k]+S2[k])
                pbgk = (bg4[k]-bg1[k]) / (bg1[k]+bg2[k])
                
                gsumf += gck - gbgk
                psum += (S4[k]-S1[k]) / (S1[k]+S2[k]) - (bg4[k]-bg1[k]) / (bg1[k]+bg2[k])
                bfsum += 0.5 * (S1[k]+S2[k])
                psum += pk - pbgk

            self.phase_data = psum / Num
            self.bf_data = bfsum / Num

            # Stockage des résultats dans les attributs de la classe (self)
            gfmoy = gsumf / Num
            self.gamma = gfmoy 
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

    # --- Méthodes d'affichage  ---
    
    def clear_slider(self):
        if self.slider_widget is not None:
            self.layout.removeWidget(self.slider_widget)
            self.slider_widget.deleteLater()
            self.slider_widget = None
        
    def display_retard(self):
        self.clear_slider()
        if self.retard_data is not None:
            show_retard(self.im_view, self.retard_data, self.info_label)

    def display_azimut(self):
        self.clear_slider()
        if self.azimuth_data is not None:
            show_azimut(self.im_view, self.azimuth_data, self.info_label)

    def display_hsv(self):
        self.clear_slider()
        if self.rgb_data is not None:
            show_hsv(self.im_view, self.rgb_data, self.info_label)
            
    def display_phase(self):
        self.clear_slider()
        if self.phase_data is not None:
            show_phase(self.im_view, self.phase_data, self.info_label)

    def display_BF(self):
        self.clear_slider()
        if self.bf_data is not None:
            show_BF(self.im_view, self.bf_data, self.info_label)

    def stat(self):
        self.clear_slider()
        statistique(self.im_view, self.info_label)

    def gaussien(self):
        self.clear_slider()
        gaussien(QtWidgets.QMainWindow,self.im_view, self.info_label)

    def median(self):
        self.clear_slider()
        median(QtWidgets.QMainWindow,self.im_view, self.info_label)
        
    def vectoriel(self):
        self.clear_slider()
        self.vec = vectorial(QtWidgets.QMainWindow,self.retard_data, self.azimuth_data, self.gamma,self.im_view, self.info_label)
        
    def tout(self):
        if self.retard_data is None:
            return
        self.clear_slider()
        self.slider_widget = SliderBlend(          
            self.retard_data, self.azimuth_data, self.phase_data,
            self.im_view, self.info_label, parent=self
        )
        self.layout.addWidget(self.slider_widget)
    
    def rideau(self):
        if self.rgb_data is None:
            return
        self.clear_slider()
        self.slider_widget = SliderHsvPhase(      
            self.rgb_data, self.phase_data,
            self.im_view, self.info_label, parent=self
        )
        self.layout.addWidget(self.slider_widget)
        
    def scl(self):
        ScaleBar(self.info_label)

# Point d'entrée pour tester le module tout seul
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    gui = MultimodalWindow()
    gui.show()
    sys.exit(app.exec())