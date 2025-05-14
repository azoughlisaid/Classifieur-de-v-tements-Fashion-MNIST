# ========== Importer les bibliothèques ==========
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

# ========== Titre de l'application ==========
st.title("👕 Fashion MNIST - Projet Deep Learning")
st.write("Bienvenue dans mon application de classification d'images ! 🚀")

# ========== Sidebar ==========
st.sidebar.title("📂 Menu")
st.sidebar.info("Cette application permet de classifier les vêtements du dataset Fashion-MNIST grâce à un modèle CNN.")

# ========== Charger le modèle ==========
try:
    model = load_model('fashion_mnist_mini_vgg_model.h5')  # <-- ton modèle ici
    st.success("✅ Modèle chargé avec succès ! Prêt à faire des prédictions.")
except Exception as e:
    st.error(f"❌ Fichier modèle non trouvé ou erreur lors du chargement : {e}")

# ========== Liste des classes ==========
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ========== Upload de l'image ==========
st.header("📤 Importer une image pour prédire sa catégorie")

uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    img = Image.open(uploaded_file).convert('L')  # 1. Niveau de gris
    img = ImageOps.autocontrast(img)              # 2. Contraste automatique
    img = img.filter(ImageFilter.MedianFilter(size=5))  # 3. Filtre anti-bruit
    img = img.resize((28, 28))                     # 4. Redimensionner

    # Normaliser et préparer pour prédiction
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Afficher l'image après traitement
    st.image(img, caption="🖼️ Image prétraitée", use_container_width=True)

    # ========== Bouton Prédire ==========
    if st.button('🎯 Prédire la catégorie'):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = round(100 * np.max(prediction), 2)

        st.success(f"✅ L'image est classée comme : **{predicted_class}** avec une confiance de **{confidence}%**.")

        # ========== Graphique des probabilités ==========
        fig, ax = plt.subplots()

        # Couleurs personnalisées
        colors = ['skyblue', 'orange', 'green', 'red', 'purple', 
                  'pink', 'brown', 'cyan', 'gold', 'gray']

        ax.barh(class_names, prediction[0], color=colors)
        ax.set_xlabel('Probabilité')
        ax.set_title('Distribution des classes prédites')
        ax.grid(True)

        st.pyplot(fig)
