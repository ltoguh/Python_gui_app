import sys
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

# --- LES IMPORTS DE TES MODULES ---
# On importe la CLASSE que l'on a créée précédemment
from biref_poo import BirefringenceWindow 
from phase_poo import PhaseWindow 
from multimodal_poo import MultimodalWindow 


class OptimagMain(QtWidgets.QMainWindow):
    
    # Constructeur
    def __init__(self):
        super().__init__()
        
        # Configuration de la fenêtre principale
        self.setWindowTitle("OPTIMAG - Menu")
        self.resize(400, 300)
        self.move(700, 400)
        
        self.init_ui() # appel de la méthode init_ui qui ajoute le contenu

    def init_ui(self):
        # Widget central et Layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        self.pixmap = QtGui.QPixmap('optimag.png')
        self.label_logo = QtWidgets.QLabel()
        self.label_logo.setPixmap(self.pixmap)
        layout.addWidget(self.label_logo)

        # Barre de boutons
        btn_bar = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_bar)

        # Zone d'affichage image (pour l'ouverture simple)
        self.im_view = pg.ImageView()
        layout.addWidget(self.im_view)
        self.im_view.hide()

        # Liste des boutons et leurs fonctions
        # Note : Pour la biréfringence, on appelle une méthode de la classe
        buttons = [
            ("Biréfringence", self.ouvrir_birefringence),
            ("Phase", self.ouvrir_phase),
            ("Multimodal", self.ouvrir_multimodal),
            ("Ouvrir", self.open_file)
        ]

        for label, fn in buttons:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(fn)
            btn_bar.addWidget(btn)
        
        self.show()

    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "", "Images (*.png *.jpg *.tiff *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            self.im_view.setImage(img)
            self.im_view.show()
            self.label_logo.hide()

    def ouvrir_birefringence(self):

        # On l'attache à self pour éviter qu'elle ne disparaisse immédiatement
        self.biref_dialog = BirefringenceWindow()
        self.biref_dialog.show()
        
    def ouvrir_phase(self):

        self.phase_dialog= PhaseWindow()
        self.phase_dialog.show()
        
    def ouvrir_multimodal(self):

        self.multimodal_dialog = MultimodalWindow()
        self.multimodal_dialog.show()

# --- LANCEMENT DE L'APPLICATION,  ne s'exécute que si on lance ce programme en tant que tel (main) ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Configuration globale pyqtgraph
    pg.setConfigOptions(imageAxisOrder='row-major')
    
    # Création et affichage du menu
    main_gui = OptimagMain()
    main_gui.show()
    
    sys.exit(app.exec())