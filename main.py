import streamlit as st
import pandas as pd
import base64
from streamlit_option_menu import option_menu 
from backend import Preprocessing, LSTM_Model

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sentimen Gojek",
    page_icon="🛵",
    layout="centered", 
    initial_sidebar_state="collapsed" 
)

# --- 2. Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f2f2f2; 
        color: #1C1C1C;
    }
    
    [data-testid="collapsedControl"], [data-testid="stSidebar"], [data-testid="stHeader"] {display: none;}
    
    .block-container {
        padding-top: 2rem !important;
        max-width: 800px;
    }
    
    /* Card Style */
    .css-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #EAEAEA;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 25px;
    }
    
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1C1C1C !important;
        border: 1px solid #D0D0D0 !important;
    }

    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-bottom: 20px;
    }
    .header-container img {
        width: 120px;
        margin-right: 20px;
    }

    /* Button Style */
    .stButton>button {
        background-color: #00AA13;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00880C;
        box-shadow: 0 4px 8px rgba(0,170,19,0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Inisialisasi Session State ---
if 'model_lstm' not in st.session_state:
    st.session_state['model_lstm'] = LSTM_Model()

# --- 3. Header & Penjelasan Sistem ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    img_base64 = get_base64_image("logogojek.png")
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{img_base64}" alt="Gojek Logo">
        <h1 style="font-size: 26px;">Sistem Analisis Sentimen LSTM</h1>
    </div>
    """, unsafe_allow_html=True)
except:
    st.title("🛵 Sistem Analisis Sentimen LSTM")

st.markdown("""
<div style="background-color: #f4fbf5; border: 1px solid #cce8d1; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
    <h4 style="margin-top: 0; color: #00AA13; font-size: 16px; font-weight: 700;">Tentang Sistem</h4>
    <p style="margin: 0; color: #444444; font-size: 14px; line-height: 1.6;">
        Sistem Analisis Sentimen ini dibangun menggunakan pemodelan <b>Long Short-Term Memory (LSTM)</b> untuk mengklasifikasikan ulasan pengguna layanan Gojek ke dalam dua kategori: <b>Positif</b> dan <b>Negatif</b>. Antarmuka ini dirancang untuk memfasilitasi pengujian kinerja model AI serta melakukan prediksi sentimen teks secara <i>real-time</i>.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 4. Navigasi ---
selected = option_menu(
    menu_title=None, 
    options=["Pelatihan & Evaluasi", "Prediksi Sentimen"],
    icons=["cloud-upload", "chat-left-text"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"border-radius": "100px", "padding": "5px", "background-color": "#FFFFFF", "border": "1px solid #EAEAEA"},
        "nav-link-selected": {"background-color": "#00AA13", "border-radius": "100px"}
    }
)

# =========================================================
# HALAMAN 1: PELATIHAN & EVALUASI
# =========================================================
if selected == "Pelatihan & Evaluasi":
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Unggah Dataset")
    uploaded_file = st.file_uploader("Pilih file dataset (.csv/.xlsx)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            if 'sentimen' not in df.columns or 'text_cleaning' not in df.columns:
                st.error("⚠️ Kolom 'sentimen' atau 'text_cleaning' tidak ditemukan!")
            else:
                if isinstance(df['sentimen'], pd.DataFrame):
                    data_sentimen = df['sentimen'].iloc[:, -1].astype(str).str.strip().str.lower()
                else:
                    data_sentimen = df['sentimen'].astype(str).str.strip().str.lower()

                total_pos = (data_sentimen == 'positive').sum()
                total_neg = (data_sentimen == 'negative').sum()
                
                if total_pos == 0 and total_neg == 0:
                    total_pos = (df['label_encoded'] == 1).sum() 
                    total_neg = (df['label_encoded'] == 0).sum()

                st.success(f"Dataset {uploaded_file.name} berhasil dimuat dengan total {len(df)} data.")
                
                if st.button("Jalankan Proses & Evaluasi Model"):
                    with st.spinner('Memproses Preprocessing & Training...'):
                        lstm = st.session_state['model_lstm']
                        metrics = lstm.evaluate_model(df, 'text_cleaning', 'sentimen')
                        
                        if metrics:
                            st.subheader("Hasil Pengujian LSTM")
                            sample_df = df.sample(min(len(df), 20)).copy()
                            
                            # Jalankan prediksi
                            preds = sample_df['text_cleaning'].apply(lambda x: lstm.predict_single(str(x)))
                            sample_df['Prediksi AI'] = preds.apply(lambda x: "😊 Positif" if x[0] == "Positif" else "😡 Negatif")
                            
                            # Jadikan string agar bisa diwarnai teksnya
                            sample_df['Keyakinan'] = preds.apply(lambda x: f"{x[1] * 100:.1f}%")
                            
                            # --- FUNGSI PEWARNAAN TABEL (PANDAS STYLER) ---
                            def color_row(row):
                                if 'Positif' in row['Prediksi AI']:
                                    color, bg = '#00AA13', '#f4fbf5'
                                else:
                                    color, bg = '#dc3545', '#fffafb'
                                # Terapkan warna hanya ke kolom Hasil Prediksi dan Keyakinan
                                return [''] * 2 + [f'color: {color}; font-weight: bold; background-color: {bg}'] * 2

                            df_display = sample_df[['text_cleaning', 'sentimen', 'Prediksi AI', 'Keyakinan']]
                            styled_df = df_display.style.apply(color_row, axis=1)
                            
                            st.dataframe(
                                styled_df,
                                column_config={
                                    "text_cleaning": st.column_config.TextColumn("Teks Ulasan", width="large"),
                                    "sentimen": "Label Asli",
                                    "Prediksi AI": "Hasil Prediksi",
                                    "Keyakinan": "Tingkat Keyakinan"
                                },
                                hide_index=True, use_container_width=True
                            )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            

                            c1, c2 = st.columns(2)
                            c1.markdown(f"""<div style="background-color: #f4fbf5; border-left: 5px solid #00AA13; padding: 15px; border-radius: 8px; border: 1px solid #cce8d1;">
                                <h4 style="margin:0; font-size:13px; color:#666;">Total Data Positif</h4>
                                <h2 style="margin:0; color: #00AA13;">{total_pos}</h2>
                            </div>""", unsafe_allow_html=True)
                            
                            c2.markdown(f"""<div style="background-color: #fffafb; border-left: 5px solid #dc3545; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                                <h4 style="margin:0; font-size:13px; color:#666;">Total Data Negatif</h4>
                                <h2 style="margin:0; color: #dc3545;">{total_neg}</h2>
                            </div>""", unsafe_allow_html=True)

                            st.success(f" Analisis selesai.")
                            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HALAMAN 2: PREDIKSI MANUAL (REVISI)
# =========================================================
else:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.markdown('<div class="css-card" style="height: 120px;">', unsafe_allow_html=True)
        st.subheader("Input Teks Ulasan")
        user_input = st.text_area(
            "Ketik ulasan di bawah ini untuk melihat prediksi:", 
            height=180, 
            placeholder="Contoh: Aplikasi ini sangat membantu saya saat butuh transportasi cepat.", 
            key="input_area",
        )

        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_out:
        st.markdown('<div class="css-card" style="height: 30px;">', unsafe_allow_html=True)
        st.subheader("Hasil Prediksi")

        if 'user_text' in st.session_state:
            res, conf = st.session_state['model_lstm'].predict_single(st.session_state['user_text'])
            emo = "😊" if res == "Positif" else "😡"
            bg = "#f8fff9" if res == "Positif" else "#fffafb"
            border = "#28a745" if res == "Positif" else "#dc3545"
            
            st.markdown(f"""
            <div class="css-card" style="background-color: {bg}; border-left: 8px solid {border};">
                <h3 style="margin:0; font-size: 40px;">{emo}</h3>
                <h2 style="margin:10px 0 0 0; color: {border};">{res}</h2>
                <p style="color:#666; font-size:14px; margin-top:5px;">Tingkat Keyakinan: <b>{conf*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; height: 180px; border: 2px dashed #EAEAEA; border-radius: 12px; color: #AAA; text-align: center; padding: 20px;">
                Hasil analisis sentimen akan muncul di sini setelah Anda menekan tombol di bawah.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tombol
    st.markdown("<div style='margin-top: -10px;'>", unsafe_allow_html=True)
    tombol = st.button("Proses Analis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if tombol:
        if user_input:
            st.session_state['user_text'] = user_input
            st.rerun() # Refresh untuk menampilkan hasil di box kanan
        else:
            st.toast("⚠️ Silakan ketik ulasan terlebih dahulu!")