import numpy as np
import sys
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter, median_filter
from PIL import Image
import matplotlib.pyplot as plt

def statistique(view, label):
    global current_data
    
    current_img = view.image

    # Il faut modifier l'image car cv2 travaille avec du uint8, pour avoir la colormap HSV on utilise aussi hsv.Je peux faire le crop sur les datas brutes mais colormap noir/blanc
    display_img = current_img - current_img.min()
    display_img = (display_img / display_img.max() * 255).astype(np.uint8)
    display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_BONE)

    rect = cv2.selectROI("Selectionnez la zone + ENTREE", display_img, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = rect

    if w > 0 and h > 0:
        crop_image = current_img[y:y+h, x:x+w]
        std_val  = np.std(crop_image)
        mean_val = np.mean(crop_image)

        # Mise à jour image
        view.setImage(crop_image)
        label.setText(f"Zoom Statistique : {w}x{h} px | Mean: {mean_val:.4f} | STD: {std_val:.4f}")

        # Label STD sur l'image
        text = pg.TextItem(html=f'<div style="text-align: center"><span style="color: #FFF; font-size: 14pt;">'
                                f'STD: {std_val:.4f}</span></div>', 
                           anchor=(0.5, 0.5))
        text.setPos(w / 2, h / 2)
        view.getView().addItem(text)
        
        
def gaussien(parent_window, view, label):  
    current_img = view.image  
    if current_img is None: return
    sig, ok = QtWidgets.QInputDialog.getDouble(parent_window, 'Filtre Gaussien', 'Sigma:', 1.0, 0.1, 50.0, 2)  
    if ok:  
        img_to_blur = current_img 
        blurred = gaussian_filter(img_to_blur, sigma=sig)
        view.setImage(blurred)  
        label.setText(f"Filtre Gaussien (sigma={sig})")

def median(parent_window, view, label):  
    current_img = view.image  
    if current_img is None: return
    noy, ok = QtWidgets.QInputDialog.getInt(parent_window, 'Filtre médian', 'Taille du noyau (impair):', 3, 1, 51, 2)  #valeur défaut, min, max, pas clic
    if ok:  
        img_to_blur = current_img 
        filtered = median_filter(img_to_blur, size=noy)
        view.setImage(filtered)  
        label.setText(f"Filtre médian (noyau={noy})")
        
def binning(parent_window, view, label):
    current_img = view.image  
    if current_img is None: return
    bin_factor, ok = QtWidgets.QInputDialog.getInt(parent_window, 'Binning', 'Taille (entier):', 4, 1, 100, 1)
    if ok:
        h, w = current_img.shape[:2]
        bh, bw = h // bin_factor, w // bin_factor

        # recadrer si nécessaire
        img_cropped = current_img[:bh*bin_factor, :bw*bin_factor]

        # reshape + moyenne
        view.setImage(img_cropped.reshape(bh, bin_factor,bw, bin_factor,-1).mean(axis=(1, 3)))
        

class BinningDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration du Binning")
        
        # Layout principal vertical
        self.main_layout = QtWidgets.QVBoxLayout(self)
        
        # 1. On appelle la méthode pour les champs de saisie
        self.init_ui()
        
        # 3. On appelle la méthode pour les boutons
        self.add_buttons()
        
        self.setStyleSheet(""" 

       QPushButton { background-color: #FFFFFF;
                    color: #1F1F1F;
                    border:none;
                    border-radius: 4px;
                    font: 20px;
                    }
       QPushButton:hover { background-color: #535454; }
       QPushButton:pressed { background-color: #21AEFC; }

       QLabel { color: #DEDEDE;
               font: 16px;
               }
       QDialog { background-color: #1F1F1F;
               font: 16px;
               }

           """)

    def init_ui(self):
        """Crée et organise les champs de saisie de données."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)

        self.spin_x = QtWidgets.QSpinBox()
        self.spin_x.setRange(1, 100)
        self.spin_x.setValue(4)

        self.spin_y = QtWidgets.QSpinBox()
        self.spin_y.setRange(1, 100)
        self.spin_y.setValue(4)
        
        self.spin_z = QtWidgets.QSpinBox()
        self.spin_z.setRange(1, 100)
        self.spin_z.setValue(4)

        layout.addRow("Binning :", self.spin_x)
        layout.addRow("Longueur (en bloc) :", self.spin_y)
        layout.addRow("Épaisseur :", self.spin_z)
        
        self.main_layout.addWidget(container)

    def add_buttons(self):
        """Crée la barre de boutons Ok / Annuler et gère les connexions."""
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        
        # Connexions aux slots standards de QDialog
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.main_layout.addWidget(buttons)

    def getValues(self):
        """Retourne les valeurs saisies par l'utilisateur."""
        return self.spin_x.value(), self.spin_y.value(), self.spin_z.value()
            

def vectorial(parent_window, central_widg, im1, im2, im3,im4, view, label):
    
    dialog = BinningDialog(parent_window)
    if dialog.exec():
        bin_factor, L, W = dialog.getValues()
        ok = True
    else:
        ok = False
    
    if ok:
        az_rad = np.deg2rad(im2 * 2)  # ×2 car azimut = angle non orienté [−90, 90]
        
        def bin_image(img, factor):
            h, w = img.shape[:2]
            img = img[: h // factor * factor, : w // factor * factor]
            if img.ndim == 2:
                return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)) #h // factor: nb de groupes de lignes, factor: nb de lignes par groupe. moyenne sur chaque bloc
            return img.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))

        
        cos_b = bin_image(np.cos(az_rad), bin_factor)
        sin_b = bin_image(np.sin(az_rad), bin_factor)
        
        angle_final_deg = np.rad2deg(np.arctan2(sin_b, cos_b)) / 2
        gamma_abs = np.abs(bin_image(im3, bin_factor))

    # 1. Création du symbole "segment" (une ligne verticale)
        path = QtGui.QPainterPath()
        path.moveTo(-0.5, -0.)
        path.lineTo(0.5, 0.)
    
        # 2. Préparation des données
        h, w = cos_b.shape
        x_coords = (np.meshgrid(np.arange(w), np.arange(h))[0] * bin_factor + bin_factor/2).flatten() #calcul de la position du centre des vecteurs. + bin_factor/2 permet de se placer au centre du bloc
        y_coords = (np.meshgrid(np.arange(w), np.arange(h))[1] * bin_factor + bin_factor/2).flatten()
        angles = angle_final_deg.flatten()
        alphas = (gamma_abs.flatten() / (gamma_abs.max() + 1e-9) * 255).astype(np.uint8)
    
        # 3. Création des pinceaux (Pens) avec Alpha
        pens = []
        for a in alphas:
            pens.append(pg.mkPen(color=(0, 126, 255, a), width=W))
    
        # 4. Création des symboles orientés
        # On doit transformer le path original pour chaque angle
        symbols = []
        for ang in angles:
            tr = QtGui.QTransform()
            tr.rotate(-ang) # Rotation horaire/anti-horaire selon le repère
            symbols.append(tr.map(path))
    
        # 5. Application au ScatterPlotItem
        v_overlay = pg.ScatterPlotItem(
            x=x_coords,
            y=y_coords,
            pen=pens,
            symbol=symbols,
            size=bin_factor * L, #longueur des traits
            pxMode=False # Important pour que 'size' soit en unités de coordonnées (pixels image)
        )
        
        view.setColorMap(pg.colormap.get('gray', source='matplotlib'))
        view.setImage(im4)
        view.getView().addItem(v_overlay)
        label.setText(f"Image multimodale avec un binning {bin_factor}x{bin_factor} pixels, un trait de longueur {L*bin_factor} pixels")
        
        # 1. Créer un layout horizontal spécifique pour les boutons
        layout_boutons = QtWidgets.QHBoxLayout()
        #layout_boutons2 = QtWidgets.QHBoxLayout()
        
        # 2. Créer les boutons
        btn_masquer = QtWidgets.QPushButton("Masquer")
        btn_afficher = QtWidgets.QPushButton("Afficher")
        
        btn_bin = QtWidgets.QPushButton("Modifier")
        #btn_L = QtWidgets.QPushButton("Longueur")
        #btn_W = QtWidgets.QPushButton("Epaisseur")
        
        # 3. Ajouter les boutons au layout HORIZONTAL
        layout_boutons.addWidget(btn_masquer)
        layout_boutons.addWidget(btn_afficher)
        
        layout_boutons.addWidget(btn_bin)
        #layout_boutons2.addWidget(btn_L)
        #layout_boutons2.addWidget(btn_W)
        
        # 4. Ajouter ce layout horizontal dans le layout VERTICAL de la fenêtre
        central_widg.layout().addLayout(layout_boutons)
        #central_widg.layout().addLayout(layout_boutons2)
        
        # 5. Connecter les actions
        btn_masquer.clicked.connect(lambda: v_overlay.hide())
        btn_afficher.clicked.connect(lambda: v_overlay.show())
        #btn_bin.clicked.connect()

def norm(data, vmin=None, vmax=None):
    if vmin is None: vmin = np.percentile(data, 1)
    if vmax is None: vmax = np.percentile(data, 99)
    return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)

class SliderBlend(QtWidgets.QWidget):
    def __init__(self, im1, im2, im3, view, label, parent=None):
        super().__init__(parent)
        self.images = {
            "hot":  (im1, "Retard"),
            "hsv":  (im2, "Azimut"),
            "gray": (im3, "Phase"),
        }
        self.view  = view
        self.label = label
        self.cmaps = list(self.images.keys())

        # Pré-calcul des images colormappées (fait une seule fois)
        self.rgb = {
            cmap: plt.get_cmap(cmap)(norm(data))[:, :, :3]
            for cmap, (data, _) in self.images.items()
        }

        layout = QtWidgets.QFormLayout(self)
        self.sliders = {}
        for cmap, (_, label_text) in self.images.items():
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.valueChanged.connect(self._update)
            layout.addRow(f"Opacité {label_text}", s)
            self.sliders[cmap] = s

        self._update()  # affichage initial

    def _update(self):
        weights = {cmap: s.value() / 100 for cmap, s in self.sliders.items()}
        total = sum(weights.values()) + 1e-9

        combined = sum(self.rgb[cmap] * w for cmap, w in weights.items()) / total
        self.view.setImage((np.clip(combined, 0, 1) * 255).astype(np.uint8))

        parts = "  |  ".join(
            f"{name} {weights[cmap]:.0%}"
            for cmap, (_, name) in self.images.items()
        )
        self.label.setText(parts)
        
class SliderHsvPhase(QtWidgets.QWidget):
    def __init__(self, im1, im2, view, label, parent = None):
        super().__init__(parent)
        self.image1 = im1
        self.image2 = im2
        
        self.view  = view
        self.label  = label
        label.setText("Comparaison images HSV et phase")
        
        layout = QtWidgets.QFormLayout(self)
        
        self.s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s.setRange(0, 100)
        self.s.setValue(50)
        self.s.valueChanged.connect(self._update)
        layout.addRow("Position du curseur", self.s)
        
        im1 = self.image1
        image2_norm = self.norm_01(self.image2, -0.1, 0.1)
        im2 = np.stack([image2_norm]*3, axis=-1) #créer 3 canaux gris/gris/gris pour matcher le RGb de l'image HSV
        
        self.view.setImage(np.concatenate((im1[:, :int(0.5*self.image1.shape[1])], im2[:, int(0.5*self.image1.shape[1]):]), axis = 1))
        
    def norm_01(self, data, vmin=None, vmax=None):
        # Si on ne donne pas de bornes, on utilise les percentiles (auto)
        if vmin is None: vmin = np.percentile(data, 1)
        if vmax is None: vmax = np.percentile(data, 99)
        
        # On applique la normalisation sur la plage choisie
        return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)
        
        
    def _update(self):
        weight = self.s.value() / 100
        pos  = int(weight * self.image1.shape[1]) #position du rideau
    
        im1 = self.image1
        image2_norm = self.norm_01(self.image2, -0.1, 0.1)
        im2 = np.stack([image2_norm]*3, axis=-1) #créer 3 canaux gris/gris/gris pour matcher le RGb de l'image HSV
    
        self.combined = np.concatenate((im1[:, :pos], im2[:, pos:]), axis = 1)
        self.view.setImage((np.clip(self.combined, 0, 1) * 255).astype(np.uint8))
        
class ScaleBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []

    def mousePressEvent(self, event):
        pos = event.position()
        self.points.append((int(pos.x()), int(pos.y())))
        self.repaint()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setBrush(QtGui.QColor(255, 0, 0))
        for point in self.points:
            p.drawEllipse(point[0], point[1], 15, 15)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dessin = ScaleBar()
    dessin.setWindowTitle("Scale Bar")
    dessin.resize(600, 400)
    dessin.show()                          
    sys.exit(app.exec())    
        
        
    
    
    
    
    
    
