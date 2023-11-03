import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np

#Tkinter ile arayüz oluşturma
root = tk.Tk()
root.title("Nesne Tanıma ve Sınıflandırma Uygulaması")
root.geometry("350x350")

# Modeli yükle
model = load_model('Model/keras_model.h5')

#Sınıf etiketlerini yükleme
with open('Model/labels.txt', 'r') as file:
    labels = file.read().splitlines()

def select_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((224, 224))  # Çıktı resminin boyutunu belirli bir boyutta sabit tutma
        np_image = np.array(image)
        np_image = np.expand_dims(np_image, axis=0)
        np_image = np_image / 255.0  #Resim piksel değerlerini normalize etme

        image = ImageTk.PhotoImage(image)
        panel = tk.Label(root, image=image)
        panel.image = image
        panel.grid(row=1, column=0, columnspan=2, padx=50, pady=20, sticky="nsew")  #Ortalamak için padx ve pady ekleme

        #Resmi tahmin etme
        predictions = model.predict(np_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]  #Tahminin doğruluk değeri

        # Tahmin edilen sınıf etiketini ve doğruluk değerini göster
        result_text = "Tahmin Sınıfı: " + predicted_label + "\nTahmin Skoru: {:.3f}".format(confidence)
        label.config(text=result_text, font=("Arial", 14))  # Metni küçültme
        label.grid(row=2, column=0, columnspan=2)  # Metni ortalamak için columnspan ekleme

select_and_predict_button = tk.Button(root, text="Resim Seç ve Tahmin Et", command=select_and_predict_image)
select_and_predict_button.grid(row=0, column=0, columnspan=2)

label = tk.Label(root, text="", font=("Arial", 14))
label.grid(row=3, column=0, columnspan=2)  #Metni ortalamak için columnspan ekleme

# Arayüz bileşenlerini ortalamak için yapılandırma
root.grid_rowconfigure(1, weight=1)  # Çıktı resmi için satırı ortalama
root.grid_columnconfigure(0, weight=1)  # Çıktı resmi için sütunu ortalama
root.grid_rowconfigure(2, weight=1)  # Metin için satırı ortalama
root.grid_columnconfigure(0, weight=1)  # Metin için sütunu ortalama

root.mainloop()
