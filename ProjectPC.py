import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Pengolahan Citra Digital",
    page_icon="🖼️",
    layout="wide"
)

# Fungsi-fungsi pengolahan citra
def biner(image, threshold):
    """Membuat citra biner dari citra berdasarkan nilai ambang (threshold)"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            if image[i, j] < threshold:
                result[i, j] = 0
            else:
                result[i, j] = 255
    
    return result

def negatif(image):
    """Membuat citra negatif dari citra"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            result[i, j] = 255 - image[i, j]
    
    return result

def image_brightening(image, b):
    """Pencerahan citra dengan menjumlahkan setiap pixel dengan skalar b"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            temp = int(image[i, j]) + b
            
            # Clipping
            if temp < 0:
                result[i, j] = 0
            elif temp > 255:
                result[i, j] = 255
            else:
                result[i, j] = temp
    
    return result

def addition(imageA, imageB):
    """Menjumlahkan dua buah citra A dan B menjadi citra baru"""
    M, N = imageA.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            temp = int(imageA[i, j]) + int(imageB[i, j])
            if temp > 255:
                result[i, j] = 255
            else:
                result[i, j] = temp
    
    return result

def multiplication(imageA, imageB):
    """Mengalikan citra A dengan citra B menjadi citra C"""
    M, N = imageA.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    for i in range(M):
        for j in range(N):
            temp = int(imageA[i, j]) * int(imageB[i, j])
            
            # Clipping
            if temp < 0:
                result[i, j] = 0
            elif temp > 255:
                result[i, j] = 255
            else:
                result[i, j] = temp
    
    return result

def konvolusi(image, mask):
    """Mengkonvolusi citra dengan mask 3x3"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.float32)
    
    for i in range(1, M-2):
        for j in range(1, N-2):
            result[i, j] = (
                image[i-1, j-1] * mask[0, 0] +
                image[i-1, j] * mask[0, 1] +
                image[i-1, j+1] * mask[0, 2] +
                image[i, j-1] * mask[1, 0] +
                image[i, j] * mask[1, 1] +
                image[i, j+1] * mask[1, 2] +
                image[i+1, j-1] * mask[2, 0] +
                image[i+1, j] * mask[2, 1] +
                image[i+1, j+1] * mask[2, 2]
            )
    
    # Normalisasi ke range 0-255
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def histogram(image):
    """Menghitung histogram citra"""
    hist = np.zeros(256)
    M, N = image.shape
    total_pixels = M * N
    
    for i in range(M):
        for j in range(N):
            hist[image[i, j]] += 1
    
    # Normalisasi histogram
    hist = hist / total_pixels
    return hist

def perataan_histogram(image):
    """Mengubah citra dengan melakukan perataan histogram (histogram equalization)"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)
    
    # Hitung histogram citra
    hist = histogram(image)
    
    # Hitung histogram hasil perataan
    hist_eq = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        sum_val = 0.0
        for j in range(i + 1):
            sum_val += hist[j]
        hist_eq[i] = int(np.floor(255 * sum_val))
    
    # Update citra sesuai histogram hasil perataan
    for i in range(M):
        for j in range(N):
            result[i, j] = hist_eq[image[i, j]]
    
    return result

def plot_histogram(image):
    """Membuat plot histogram"""
    fig, ax = plt.subplots(figsize=(6, 4))
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.set_xlim((0, 256))
    ax.set_xlabel('Intensitas Pixel')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Histogram')
    ax.grid(True, alpha=0.3)
    return fig

def convert_to_grayscale(image):
    """Konversi gambar ke grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# Sidebar untuk pilihan fitur
st.sidebar.title("🎨 Fitur Pengolahan Citra")
st.sidebar.markdown("---")

feature = st.sidebar.selectbox(
    "Pilih Fitur:",
    [
        "Original",
        "Thresholding (Binerisasi)",
        "Citra Negatif",
        "Image Brightening",
        "Convolution (Konvolusi)",
        "Histogram Equalization"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("📝 Pilih fitur di atas untuk mengolah gambar yang Anda upload.")

# Header utama
st.title("🖼️ Aplikasi Pengolahan Citra Digital")
st.subheader("Kelompok 3 :")
st.markdown("Muhammad Ruhul Jadid (123230046)")
st.markdown("Ardhika Rizki Akbar Pratama (123230057)")
st.markdown("Brian Zahran Putra (123230195) ")
st.markdown("Ikhsan Fillah Hidayat (123230219)")
st.markdown("---")

# Upload gambar
uploaded_file = st.file_uploader(
    "📁 Upload Gambar (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop gambar atau klik untuk browse"
)

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    # Simpan gambar asli untuk ditampilkan di input
    rgb_image = image_array.copy()
    gray_image = convert_to_grayscale(image_array)
    # Variabel untuk gambar output
    output_image = gray_image.copy()

    # Parameter untuk fitur yang membutuhkan input
    st.markdown("---")

    if feature == "Thresholding (Binerisasi)":
        st.subheader("⚙️ Parameter Thresholding")
        threshold_value = st.slider("Nilai Threshold (0-255):", 0, 255, 127)
        output_image = biner(gray_image, threshold_value)

    elif feature == "Citra Negatif":
        output_image = negatif(gray_image)

    elif feature == "Image Brightening":
        st.subheader("⚙️ Parameter Brightening")
        brightness_value = st.slider("Nilai Brightness (-100 sampai 100):", -100, 100, 50)
        output_image = image_brightening(gray_image, brightness_value)


    elif feature == "Convolution (Konvolusi)":
        st.subheader("⚙️ Parameter Konvolusi")
        mask_type = st.selectbox(
            "Pilih Jenis Filter/Mask:",
            [
                "Smoothing (Average)",
                "Gaussian Blur",
                "Sharpening",
                "Edge Detection (Sobel X)",
                "Edge Detection (Sobel Y)",
                "Edge Detection (Laplacian)"
            ]
        )
        masks = {
            "Smoothing (Average)": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
            "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "Sharpening": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "Edge Detection (Sobel X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "Edge Detection (Sobel Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            "Edge Detection (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        }
        selected_mask = masks[mask_type]
        output_image = konvolusi(gray_image, selected_mask)

    elif feature == "Histogram Equalization":
        output_image = perataan_histogram(gray_image)

    elif feature == "Original":
        output_image = rgb_image.copy()

    # Tampilkan hasil dalam dua kolom: input dan output
    st.markdown("---")
    st.header("📊 Hasil Pengolahan")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(rgb_image, width=200, clamp=True)
        st.subheader("📈 Histogram Input")
        fig_input = plot_histogram(gray_image)
        st.pyplot(fig_input)
        plt.close()
    with col2:
        st.subheader("🖼️ Gambar Output")
        if feature == "Original":
            st.image(output_image, width=200, clamp=True)
            st.subheader("📈 Histogram Output")
            fig_output = plot_histogram(gray_image)
            st.pyplot(fig_output)
            plt.close()
        else:
            st.image(output_image, width=200, clamp=True)
            st.subheader("📈 Histogram Output")
            fig_output = plot_histogram(output_image)
            st.pyplot(fig_output)
            plt.close()
    # Download hasil
    st.markdown("---")
    st.subheader("💾 Download Hasil")
    output_pil = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Gambar Output",
        data=byte_im,
        file_name=f"output_{feature.lower().replace(' ', '_')}.png",
        mime="image/png"
    )
    st.markdown("""
    ### 📋 Fitur yang Tersedia:
    1. **Thresholding (Binerisasi)** - Membuat citra biner berdasarkan nilai ambang
    2. **Citra Negatif** - Membalik intensitas pixel
    3. **Image Brightening** - Mengatur kecerahan gambar
    4. **Convolution** - Konvolusi dengan berbagai filter (smoothing, sharpening, edge detection)
    5. **Histogram Equalization** - Perataan histogram untuk meningkatkan kontras
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 Aplikasi Pengolahan Citra Digital | Dibuat dengan Streamlit"
    "</div>",
    unsafe_allow_html=True
)
