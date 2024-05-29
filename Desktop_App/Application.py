import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
import cv2
from rembg import remove
from tensorflow.keras.applications import VGG16
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Charger le modèle pré-entrainé et l'encoder pour les classes
        self.loaded_model = pickle.load(open('RidgeClassifierCV.pkl', 'rb'))
        self.le = LabelEncoder()
        self.classs = ['0', '4', '6', '11', '12', '14', '17']
        self.le.fit_transform(self.classs)

        # Initialiser l'interface utilisateur
        self.init_ui()

    def init_ui(self):
        self.setup_widgets()
        self.setWindowTitle('Interface Utilisateur pour la Classification d\'Images de trafic routier')
        self.setGeometry(400, 60, 1000, 700)
        self.setStyleSheet("background-color: #ffffff; border: 0; margin: 0;")  
        icon = QIcon("icon.png")
        self.setWindowIcon(icon)
        self.show()

    def setup_widgets(self):
        # Bouton pour importer une image
        button_import_image = QPushButton('Importer image', self)
        button_import_image.clicked.connect(self.on_button_import_image_click)
        button_import_image.setGeometry(400, 300, 200, 50)
        self.apply_hover_style(button_import_image)

        # Affichage de l'image importée
        self.label_show_image_originale = QLabel(self)
        self.label_show_image_originale.setGeometry(150, 370, 300, 300)
        self.label_show_image_originale.setScaledContents(True)

        # Affichage de la prédiction de classe
        self.label_prediction = QLabel(self)
        self.label_prediction.setGeometry(500, 460, 480, 50)
        self.label_prediction.setStyleSheet("font-size: 20px; font-weight: bold; color : #051c2b;")
        self.label_prediction.setWordWrap(True)

        # Affichage de la description de la prédiction de classe
        self.label_Description_PredictCLass = QLabel(self)
        self.label_Description_PredictCLass.setGeometry(550, 510, 400, 50)
        self.label_Description_PredictCLass.setStyleSheet("font-size: 15px; font-weight: bold; color : #051c2b;")
        self.label_Description_PredictCLass.setWordWrap(True)

        # Barre supérieure avec le titre de l'application
        topBare = QFrame(self)
        topBare.setStyleSheet("background-color: #00031b; border: 0; margin: 0;")
        topBare.setGeometry(0, 0, 1000, 90)
        label_image = QLabel("Application pour la Classification d\'Images de trafic routier", topBare)
        label_image.setGeometry(50, 20, 900, 50)
        label_image.setStyleSheet(" color: #ffffff; font-size: 30px; font-weight: bold;")

        # Section d'affichage des images de panneaux de signalisation et leurs descriptions
        imagesPPredit = QFrame(self)
        imagesPPredit.setStyleSheet("background-color:  #d8f0ff; border: 0; margin: 0;")
        imagesPPredit.setGeometry(0, 90, 1000, 200)

        label_PPredit  = QLabel("L'application permet de classer les images de panneaux de signalisation appartient à 7 classes suivantes:", imagesPPredit)
        label_PPredit.setGeometry(10, 0, 900, 50)
        label_PPredit.setStyleSheet(" color: #000000; font-size: 15px; font-weight: bold;")

        # Configuration des images de panneaux de signalisation et de leurs descriptions
        self.setup_image_labels(imagesPPredit)

    def setup_image_labels(self, imagesPPredit):
        image_infos = [
            ("Screenshot_0.png", "Speed limit (5km/h)"),
            ("Screenshot_4.png", "Speed limit (50km/h)"),
            ("Screenshot_6.png", "speed limit (70km/h)"),
            ("Screenshot_11.png", "Don't Go Left"),
            ("Screenshot_12.png", "Don't Go Left or Right"),
            ("Screenshot_14.png", "Don't overtake from Left"),
            ("Screenshot_17.png", "No horn")
        ]

        x_offset = 30
        for image_path, description in image_infos:
            label_show_image = QLabel(imagesPPredit)
            label_show_image.setGeometry(x_offset, 50, 100, 100)
            label_show_image.setScaledContents(True)
            pixmap = QPixmap(image_path)
            label_show_image.setPixmap(pixmap)
            label_show_image.setAlignment(Qt.AlignCenter)

            label_description = QLabel(description, imagesPPredit)
            label_description.setGeometry(x_offset, 150, 115, 50)
            label_description.setAlignment(Qt.AlignCenter)

            x_offset += 140

    def apply_hover_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #ffffff;
                border-radius: 2px;
                padding: 12px 10px;
            }
            QPushButton:hover {
                background-color: #1469a1;
            }
        """)
        button.setCursor(Qt.PointingHandCursor)

    def on_button_import_image_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                pixmap = QPixmap(file_paths[0])
                self.label_show_image_originale.setPixmap(pixmap)
                prediction = self.predict_class(file_paths[0])
                self.label_prediction.setText(f"Predicted Class: {prediction}")

    def predict_class(self, file_path):
        # Description des classes de panneaux de signalisation
        class_descriptions = {
            "0": "Speed limit (5km/h)",
            "4": "Speed limit (50km/h)",
            "6": "speed limit (70km/h)",
            "11": "Don't Go Left",
            "12": "Don't Go Left or Right",
            "14": "Don't overtake from Left",
            "17": "No horn"
        }

        SIZE = 256
        # Charger le modèle VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
        for layer in base_model.layers:
            layer.trainable = False

        # Charger l'image, la prétraiter et extraire les caractéristiques
        img = cv2.imread(file_path)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = remove(img)
        result = cv2.resize(result, (256, 256))
        if result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
        result = result / 255.0
        feature_extractor = base_model.predict(result[np.newaxis, ...])
        features = feature_extractor.reshape(1, -1)
        # Faire la prédiction
        prediction = self.loaded_model.predict(features)
        predicted_class = self.le.inverse_transform(prediction)[0]
        # Afficher la description de la classe prédite
        if predicted_class == '0':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique une limite de vitesse de 5 km/h")
        elif predicted_class == '4':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique une limite de vitesse de 50 km/h")
        elif predicted_class == '6':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique la fin de la limite de vitesse de 70 km/h")
        elif predicted_class == '11':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique de ne pas tourner à gauche")
        elif predicted_class == '12':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique de ne pas tourner à gauche ou à droite")
        elif predicted_class == '14':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique de ne pas dépasser par la gauche")
        elif predicted_class == '17':
            self.label_Description_PredictCLass.setText("Le panneau de signalisation indique de ne pas klaxonner")

        return f"{class_descriptions.get(predicted_class, 'Unknown')} (class: {predicted_class})"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
