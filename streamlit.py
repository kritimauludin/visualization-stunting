import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Fungsi untuk memuat data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Fungsi untuk memecah string dan mengonversi ke bulan
def convert_string_to_months(date_str):
    parts = date_str.split(' - ')
    years = int(parts[0].split(' ')[0])
    months = int(parts[1].split(' ')[0])
    days = int(parts[2].split(' ')[0])
    
    days_in_month = 30.44
    return (years * 12) + months + (days / days_in_month)

# Fungsi untuk preprocessing data
def preprocess_data(data):
    data['TB/U'] = data['TB/U'].apply(lambda x: 1 if x == 'Pendek' else 0)

    # Konversi 'Usia Saat Ukur' ke 'Usia Saat Ukur (Bulan)'
    data['Usia Saat Ukur (Bulan)'] = data['Usia Saat Ukur'].apply(convert_string_to_months)

    # Memilih kolom yang relevan
    X = data[['Tinggi', 'Usia Saat Ukur (Bulan)', 'ZS TB/U']]
    y = data['TB/U']

    # Standarisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Fungsi untuk visualisasi spasial
def spatial_visualization(data):
    # Pisahkan kolom "Desa(lat, lng)" menjadi dua kolom "lat" dan "lon"
    # data[['lat', 'lon']] = data['Desa(lat, lng)'].str.split(',', expand=True)

    # Konversi kolom lat dan lon ke tipe data float
    data['lat'] = data['Latitude'].astype(float)
    data['lon'] = data['Longitude'].astype(float)

    # Konversi kolom date menjadi format datetime
    data['date'] = pd.to_datetime(data['Tanggal Pengukuran'])

    # Filter data untuk baris di mana 'TB/U' bukan 'Normal'
    filtered_data = data[data['TB/U'] != 'Normal']

    # Hitung frekuensi kemunculan setiap kombinasi lat dan lon
    location_counts = filtered_data.groupby(['lat', 'lon', 'date']).size().reset_index(name='count')

    # Buat peta dasar dengan folium
    m = folium.Map(location=[filtered_data['lat'].mean(), filtered_data['lon'].mean()], zoom_start=12)

    # Tambahkan lingkaran ke peta untuk setiap lokasi dengan radius berdasarkan jumlah kejadian
    for _, row in location_counts.iterrows():
        folium.Circle(
            location=(row['lat'], row['lon']),
            radius=row['count'] * 10,  # Radius yang ditentukan oleh frekuensi kemunculan
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(m)

    # Tampilkan peta dalam Streamlit
    st_folium(m, width=700, height=500)

# Fungsi untuk visualisasi data
def visualize_data(X, y, X_resampled, y_resampled):
    st.subheader('Scatter Plots Before and After SMOTE')

    df_before = pd.DataFrame(X, columns=['Tinggi', 'Usia Saat Ukur (Bulan)', 'ZS TB/U'])
    df_before['Stunting'] = y

    df_after = pd.DataFrame(X_resampled, columns=['Tinggi', 'Usia Saat Ukur (Bulan)', 'ZS TB/U'])
    df_after['Stunting'] = y_resampled

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(data=df_before, x='Usia Saat Ukur (Bulan)', y='ZS TB/U', hue='Stunting', ax=axs[0])
    axs[0].set_title('Z-score TB/U vs. Age in months Before SMOTE')
    axs[0].set_xlabel('Age in months')
    axs[0].set_ylabel('Z-score TB/U')

    sns.scatterplot(data=df_after, x='Usia Saat Ukur (Bulan)', y='ZS TB/U', hue='Stunting', ax=axs[1])
    axs[1].set_title('Z-score TB/U vs. Age in months After SMOTE')
    axs[1].set_xlabel('Age in months')
    axs[1].set_ylabel('Z-score TB/U')

    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    st.pyplot(fig)

# Fungsi untuk visualisasi distribusi kelas sebelum dan sesudah SMOTE
def visualize_class_distribution(y_before_smote, y_resampled):
    st.subheader('Class Distribution Before and After SMOTE')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.countplot(x=y_before_smote, ax=axs[0])
    axs[0].set_title('Class Distribution Before SMOTE')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Count')

    sns.countplot(x=y_resampled, ax=axs[1])
    axs[1].set_title('Class Distribution After SMOTE')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Count')

    st.pyplot(fig)

# Fungsi untuk membangun dan melatih model LSTM
def build_and_train_lstm(X_train, y_train, X_val, y_val, n_units=50, dropout_rate=0.2, learning_rate=0.001, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    return model, history

# Fungsi untuk menampilkan riwayat pelatihan model
def plot_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Training and Validation Accuracy')
    
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title('Training and Validation Loss')
    
    st.pyplot(fig)

# Fungsi untuk menampilkan Confusion Matrix, F1 Score, dan AUC
def evaluate_model(model, X_val, y_val):
    st.subheader('Model Evaluation')

    # Prediksi dengan data validasi
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    # Display the confusion matrix using heatmap
    st.write('Confusion Matrix:')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot()

    # F1 Score
    f1 = f1_score(y_val, y_pred)
    st.write(f'F1 Score: {f1:.4f}')

    # AUC
    auc = roc_auc_score(y_val, y_pred_prob)
    st.write(f'AUC: {auc:.4f}')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot()

# Fungsi untuk klasifikasi status
def stunting_status(zsTbU):
    if zsTbU < -3:
        status = "Sangat pendek (severely stunted)"
    elif zsTbU > -3 and zsTbU < -2:
        status = "Pendek (stunted)"
    elif zsTbU > -2 and zsTbU < 3:
        status = "Normal"
    elif zsTbU > 3:
        status = "Tinggi"
    else :
        status = "Tidak Valid"

    return status

def countZsTbU(TB, U, gender='Laki-Laki'):
    """
    Menghitung z-score tinggi badan anak berdasarkan usia dan jenis kelamin menggunakan data WHO.
    
    :param TB: Tinggi badan anak (cm)
    :param U: Umur anak (bulan)
    :param gender: Jenis kelamin anak ('Laki-Laki' atau 'Perempuan')
    :return: Z-score untuk TB/U
    """
    # Tabel referensi WHO mean dan SD berdasarkan umur dan gender (contoh)
    who_data = {
        'Laki-Laki': {
            24: {'mean': 87.8, 'sd': 3.44}, 
            36: {'mean': 95.1, 'sd': 3.64}, 
            48: {'mean': 102.0, 'sd': 3.94} 
        },
        'Perempuan': {
            24: {'mean': 85.7, 'sd': 3.35}, 
            36: {'mean': 93.9, 'sd': 3.55}, 
            48: {'mean': 101.0, 'sd': 3.85} 
        }
    }

    # Memastikan data untuk usia dan gender yang valid tersedia
    if U not in who_data[gender]:
        st.warning("Data untuk umur tersebut tidak tersedia dalam referensi WHO.")
    
    # Ambil mean dan sd dari tabel berdasarkan usia dan gender
    if U != 0:
        mean_TB_U = who_data[gender][U]['mean']
        sd_TB_U = who_data[gender][U]['sd']

        # Hitung z-score
        z_score = (TB - mean_TB_U) / sd_TB_U
        return z_score
    else:
        z_score = 0


# Fungsi untuk deteksi dini
def detection_stunting():
    st.write('Deteksi Dini Stunting :')
    #pembagian kolom
    col1, col2 = st.columns(2)

    with col1 : 
        TB = st.number_input("Tinggi Badan (cm): ", step=1)
    with col2 : 
        U = st.number_input("Umur Anak (bulan) : ", step=1)

    with col1 : 
        gender = st.selectbox("Jenis Kelamin?", ("Laki-Laki", "Perempuan"),)
    with col2 : 
        z_score = countZsTbU(TB, U, gender)
        if z_score != 0:
            zsTbU = st.number_input("ZS TB/U : ", z_score, disabled=True)

    with col1 : 
        if st.button("Check Status") :
            checkStatus = stunting_status(float(zsTbU))
        else :
            checkStatus = False;
    with col2 :
        st.text("")
        st.text("")
        st.text("")
        
    with col1:
        st.markdown("----------------")
    with col2:
        st.markdown("----------------")

    if checkStatus != False:
        message = """
        Variabel :\n
        - Tinggi Badan Anak : """+ str(int(TB)) +""" cm\n
        - Umur Anak : """+ str(U) +""" bulan\n
        - Nilai ZS(TB/U) : """+str(round(zsTbU, 2))+"""
        """

        output = """
        Deteksi Stunting :\n
        Dengan nilai ZS(TB/U) """+str(round(zsTbU, 2))+"""\n 
        Status Anak **"""+str(checkStatus)+ """** 
        """
        with col1 : 
            st.warning(message)
        with col2 : 
            st.warning(output)
    else:
        st.warning("Klik tombol 'Check Status' untuk menampilkan hasil!")


st.title('Aplikasi Stunting Cerdas')

# Menu di sidebar
menu = st.sidebar.selectbox('Menu', ['Klasifikasi Stunting', 'Visualisasi Sebaran Zona Stunting'])

# File uploader
uploaded_file = st.sidebar.file_uploader('Upload your Excel file', type=['xlsx'])

if uploaded_file:
    data = load_data(uploaded_file)
    X, y = preprocess_data(data)

    # Terapkan SMOTE untuk semua visualisasi dan klasifikasi
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    if menu == 'Klasifikasi Stunting':
        st.header('Klasifikasi Stunting')

        # Pengaturan Hyperparameter
        n_units = st.sidebar.slider('LSTM Units', min_value=10, max_value=100, value=50)
        dropout_rate = st.sidebar.slider('Dropout Rate', min_value=0.0, max_value=0.5, value=0.2)
        learning_rate = st.sidebar.slider('Learning Rate', min_value=0.0001, max_value=0.01, value=0.001)
        epochs = st.sidebar.slider('Epochs', min_value=10, max_value=200, value=100)
        batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=128, value=32)

        # Split data untuk training dan validasi
        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Mengubah bentuk data untuk LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model, history = build_and_train_lstm(X_train, y_train, X_val, y_val, n_units, dropout_rate, learning_rate, epochs, batch_size)

        st.subheader('Training History')
        st.write('Grafik loss dan accuracy dari pelatihan model:')
        plot_training_history(history)

        # Evaluasi model
        evaluate_model(model, X_val, y_val)

    elif menu == 'Visualisasi Sebaran Zona Stunting':
        st.header('Visualization Data')
        st.write('Distribusi sebaran stunting:')
        data = load_data(uploaded_file)
        spatial_visualization(data)

        st.write('Visualisasi data stunting scatter plot:')
        visualize_data(X, y, X_resampled, y_resampled)

        st.write('Distribusi kelas sebelum dan sesudah SMOTE:')
        visualize_class_distribution(y, y_resampled)

else:
    if menu == 'Klasifikasi Stunting':
       detection_stunting()

    st.write('Please upload an Excel file to more feature.')
