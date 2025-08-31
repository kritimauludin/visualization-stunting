import pickle
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler

# Load the model
stunting_model = pickle.load(open('stunting_model.sav', 'rb'))

# Tambahkan CSS untuk menebalkan teks pada sidebar
st.markdown("""
    <style>
    .nav-link p {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Membuat sidebar dengan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # required
        options=["Beranda", "Informasi", "Visualisasi Data", "Prediksi Stunting", "Visualisasi Penyebaran"],  # required
        icons=["house", "info-circle", "bar-chart", "activity", "map"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# Tampilan berdasarkan pilihan menu
if selected == "Beranda":
    st.title("Selamat Datang Di Aplikasi Prediksi Stunting.")
    st.write("Aplikasi Prediksi Stunting adalah aplikasi untuk memprediksi kemungkinan balita mengalami stunting berdasarkan pada data rekam medik di Kota Bogor. **Pilih menu pada sidebar** untuk mencoba prediksi")

elif selected == "Informasi":
    st.title("Informasi Stunting")

    st.subheader('Pengertian')
    st.write("Stunting pada balita adalah kondisi di mana anak mengalami pertumbuhan yang terhambat sehingga tinggi badannya lebih pendek dibandingkan dengan anak-anak seusianya. Stunting disebabkan oleh kekurangan gizi kronis selama periode kritis pertumbuhan dan perkembangan, terutama dalam 5 tahun kehidupannya.")
    
    st.subheader('Faktor')
    st.write("1. Gizi Buruk: Kekurangan asupan nutrisi yang memadai selama kehamilan dan masa bayi.")
    st.write("2. Penyakit Infeksi Berulang: Penyakit seperti diare dan infeksi saluran pernapasan yang sering..")
    st.write("3. Kurangnya Sanitasi dan Kebersihan: Lingkungan yang tidak bersih dapat menyebabkan infeksi berulang.")
    st.write("4. Kurangnya Akses ke Layanan Kesehatan: Termasuk kurangnya imunisasi dan perawatan kesehatan dasar.")
    st.write("5. Faktor Sosial dan Ekonomi: Kemiskinan, kurangnya pendidikan ibu, dan kondisi sosial lainnya.")

elif selected == "Visualisasi Data":
    st.title("Visualisasi Data")

    st.subheader("Visualisasi Barchart")
    st.write("Visualisasi barchart (grafik batang) adalah representasi data dalam bentuk batang horizontal atau vertikal. Setiap batang mewakili kategori atau nilai tertentu dan panjangnya sesuai dengan nilai data. Barchart berguna untuk membandingkan berbagai kategori data atau menunjukkan perubahan data dari waktu ke waktu.")
    
    st.write("1. Barchart")
    st.image('image/barchart.png', width=700)
    st.write("Pada Barchart ini menampilkan data stunting berbentuk grafik. Dapat diketahui bahwa Balita dengan kasus pendek (stunting) lebih banyak ditemukan dibandikan kasus sangat pendek (several stunting)")

    st.subheader("Featured Correlations")
    st.write("Featured Correlations berfungsi untuk mengetahui apakah diantara dua variabel terdapat hubungan atau tidak, dan jika ada hubungan bagaimanakah arah hubungan dan seberapa besar hubungan tersebut.")
    st.image('image/featured correlations.png', width=700)
    st.write('Dapat diketahui dari gambar diatas, bahwasanya jika hasil korelasi mendekati nilai 1, maka korelasi tersebut dikatakan baik, namun jika mendekati nilai -1 maka korelasi dikatakan buruk. Dapat dilihat dari hasil visualisasi di atas, yang memiliki nilai korelasi baik terdapat pada variabel "Usia", “Berat” dan “Tinggi” yaitu memiliki nilai hasil korelasinya 0.84, 0.89 dan 0.94, adapun nilai korelasi yang buruk terdapat pada variabel “BB/U” dan "ZS BB/U", yaitu memiliki nilai -0.83, jika hasil korelasi yang berbentuk diagonal menyamping tersebut itu merupakan korelasi variabel dengan dirinya sendiri.')

    st.subheader("Categorial Plot")
    st.write('Categorical plots adalah visualisasi seaborn dengan melakukan agregasi nilai per kolom yang berbentuk kategori.')
    st.image('image/categorial plot.jpg', width=700)
    st.write('Dapat diketahui dari gambar diatas, plot dilakukan pada setiap variabel berdasarkan TB/U sehingga membentuk suatu pola persebaran data pada masing-masing variabel. Pesebaran data balita yang terdiagnosis stunting rata-rata memiliki nilai yang tinggi pada variabel-variabel tersebut. Hal tersebut selaras dengan gambar correlations features, dimana variabel tersebut memiliki nilai yang besar dalam mempengaruhi balita berpotensi stunting.')

elif selected == "Prediksi Stunting":
    st.title("Prediksi Stunting")

    # Split the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        Usia_Saat_Ukur = st.number_input('Input Usia (Bulan)', min_value=0, max_value=60, value=0, step=1)

    with col2:
        Berat = st.number_input('Input Berat (kg)', min_value=0.0, max_value=30.0, value=0.0, step=0.1)

    with col1:
        Tinggi = st.number_input('Input Tinggi (cm)', min_value=0.0, max_value=105.0, value=0.0, step=0.1)

    with col2:
        ZS_TB_U = st.number_input('Input ZS TB/U', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

    # Code for prediction
    stunting_diagnosis = ''

    # Create a button for prediction
    if st.button('Test Prediksi Stunting'):
        stunting_prediction = stunting_model.predict([[Usia_Saat_Ukur, Berat, Tinggi, ZS_TB_U]])

        if stunting_prediction[0] == 1:
            stunting_diagnosis = 'Sangat Pendek'
        else:
            stunting_diagnosis = 'Pendek'

        st.success(stunting_diagnosis)

elif selected == "Visualisasi Penyebaran":
    st.title("Visualisasi Penyebaran Stunting")

    # Fungsi untuk memuat data dari file Excel
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data

    # Fungsi untuk visualisasi spasial
    def spatial_visualization(data):
        data['lat'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['lon'] = pd.to_numeric(data['Longitude'], errors='coerce')
        data = data.dropna(subset=['lat', 'lon'])
        data['date'] = pd.to_datetime(data['Tanggal Pengukuran'], errors='coerce')
        
        filtered_data = data[data['TB/U'].isin(['Pendek', 'Sangat Pendek'])]
        location_counts = filtered_data.groupby(['lat', 'lon']).size().reset_index(name='count')
        status_counts = filtered_data.groupby(['lat', 'lon', 'TB/U']).size().unstack(fill_value=0)
        location_info = filtered_data.groupby(['lat', 'lon']).agg({
            'Desa/Kel': 'first',
            'Pukesmas': 'first'
        }).reset_index()
        
        m = folium.Map(location=[filtered_data['lat'].mean(), filtered_data['lon'].mean()], zoom_start=12)
        
        for (lat, lon), status in status_counts.iterrows():
            count_pendek = int(status.get('Pendek', 0))
            count_sangat_pendek = int(status.get('Sangat Pendek', 0))
            location_row = location_info[(location_info['lat'] == lat) & (location_info['lon'] == lon)]
            desa = location_row['Desa/Kel'].values[0]
            pukesmas = location_row['Pukesmas'].values[0]

            if count_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_pendek * 10,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m)

            if count_sangat_pendek > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=count_sangat_pendek * 10,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6
                ).add_to(m)
            
            popup_text = f"""
            Lokasi: ({lat}, {lon}) | 
            Desa: {desa} | 
            Puskesmas: {pukesmas} | 
            Pendek: {count_pendek} | 
            Sangat Pendek: {count_sangat_pendek} | 
            Total: {count_pendek + count_sangat_pendek}
            """
            folium.Marker(
                location=(lat, lon),
                popup=folium.Popup(popup_text,  max_width=450, parse_html=True),
                icon=folium.Icon(color='green')
            ).add_to(m)
        
        st_folium(m, width=700, height=500)

        st.markdown("---")

    # Fungsi untuk menampilkan diagram batang berdasarkan Tahun
    def line_chart(data):
        if 'Tahun' in data.columns and 'Kec' in data.columns:
            # Hitung jumlah total kasus per tahun
            total_per_year = data.groupby('Tahun').size().reset_index(name='Total Kasus')
            # Hitung jumlah kasus per tahun dan kecamatan
            count_data = data.groupby(['Tahun', 'Kec']).size().reset_index(name='Jumlah Kasus')
            # Gabungkan dengan total per tahun untuk menghitung persentase
            count_data = count_data.merge(total_per_year, on='Tahun')
            count_data['Persentase Kasus'] = (count_data['Jumlah Kasus'] / count_data['Total Kasus']) * 100

            # Buat diagram garis dengan warna berdasarkan Kecamatan
            fig = px.line(
                count_data, x='Tahun', y='Persentase Kasus', color='Kec',
                title='Prevalensi Stunting di Kota Bogor per Kecamatan',
                labels={'Tahun': 'Tahun', 'Persentase Kasus': 'Persentase (%)', 'Kec': 'Kec'},
                markers=True
            )
            fig.update_xaxes(tickmode='linear', tickformat='d')
            fig.update_layout(yaxis_tickformat=".2f")  
            st.plotly_chart(fig)

            pivot_table = count_data.pivot(index='Tahun', columns='Kec', values='Persentase Kasus').fillna(0)
            pivot_table['Total Kasus'] = total_per_year.set_index('Tahun')['Total Kasus'].astype(int)  
            pivot_table.index = pivot_table.index.astype(int)
            pivot_table = pivot_table.T
            pivot_table_display = pivot_table.applymap(lambda x: f"{x:.2f}%" if isinstance(x, float) else f"{x:,}")
            pivot_table_display.loc['Total Kasus'] = pivot_table.loc['Total Kasus'].apply(lambda x: f"{int(x):,}")
            st.dataframe(pivot_table_display)

        else:
            st.write("Kolom 'Tahun' atau 'Kecamatan' tidak ditemukan dalam dataset.")

        st.markdown("---")

# Fungsi untuk menampilkan diagram batang dengan filter kategori
    def bar_chart(data):
        if 'Tahun' in data.columns and 'Desa/Kel' in data.columns and 'TB/U' in data.columns:
            # Pilihan tahun
            tahun_options = sorted(data['Tahun'].unique())
            selected_tahun = st.selectbox("Pilih Tahun:", tahun_options, index=len(tahun_options)-1)

            # Filter berdasarkan tahun
            data = data[data['Tahun'] == selected_tahun]

            # Pilihan kategori
            kategori_options = ['Pendek', 'Sangat Pendek']
            selected_kategori = st.multiselect("Pilih Kategori Stunting:", kategori_options, default=kategori_options)

            # Filter berdasarkan kategori
            filtered_data = data[data['TB/U'].isin(selected_kategori)]

            if filtered_data.empty:
                st.write("Tidak ada data yang tersedia untuk filter yang dipilih.")
                return

            # Hitung jumlah kasus per Desa/Kel
            count_data = filtered_data.groupby(['Desa/Kel', 'TB/U']).size().reset_index(name='Jumlah Kasus')

            # Warna kategori
            color_map = {'Pendek': 'red', 'Sangat Pendek': 'blue'}

            # Buat diagram batang
            fig = px.bar(
                count_data,
                x='Desa/Kel',  
                y='Jumlah Kasus',  
                color='TB/U',  
                title=f'Jumlah Kasus Stunting per Desa/Kelurahan (Tahun {selected_tahun})',
                labels={'Jumlah Kasus': 'Jumlah Kasus', 'Desa/Kel': 'Desa/Kelurahan', 'TB/U': 'Kategori'},
                color_discrete_map=color_map,
                text_auto=True  
            )

            # Atur tampilan agar lebih optimal
            fig.update_xaxes(
                categoryorder="total descending",
                tickangle=-45,  
                tickfont=dict(size=6),  
                automargin=True
            )
            fig.update_layout(width=2000, height=800)  

            st.plotly_chart(fig)

        else:
            st.write("Kolom 'Tahun', 'Desa/Kel', atau 'TB/U' tidak ditemukan dalam dataset.")

# Fungsi untuk memuat data dari file Excel
    def load_data(file_path):
        return pd.read_excel(file_path)

# Inisialisasi session state jika belum ada
    if 'data' not in st.session_state:
        st.session_state['data'] = None

    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

# Upload file untuk visualisasi penyebaran
    uploaded_file = st.file_uploader("Upload file data stunting", type=['xlsx'])

# Jika pengguna mengunggah file, simpan ke session_state dan load datanya
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['data'] = load_data(uploaded_file)

# Gunakan data dari session_state jika sudah ada
    data = st.session_state['data']

# Pastikan data tidak None sebelum digunakan
    if data is not None:
        st.write("✅ File telah diunggah dan diproses!")
        spatial_visualization(data)
        line_chart(data)
        bar_chart(data)
    else:
        st.write("❌ Silakan upload file data terlebih dahulu.")