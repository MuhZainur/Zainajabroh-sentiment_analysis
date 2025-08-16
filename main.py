# main.py
import pandas as pd
import re
from transformers import pipeline
from google_play_scraper import Sort, reviews
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import library baru untuk visualisasi
import matplotlib
matplotlib.use('Agg') # <-- Penting! Gunakan backend non-interaktif
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import base64
from io import BytesIO

# ==============================================================================
# 0. Daftar Stopwords (Kata-kata umum yang akan diabaikan)
# ==============================================================================
# Sumber: https://github.com/stopwords-iso/stopwords-id/blob/master/stopwords-id.txt
STOPWORDS_ID = [
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir",
    "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara",
    "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah", "arti", "artinya", "asal",
    "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya", "bagai", "bagaikan", "bagaimana",
    "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal",
    "bakalan", "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini", "beginian", "beginikah",
    "beginilah", "begitu", "begitukah", "begitulah", "begitupun", "bekerja", "belakang", "belakangan",
    "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya",
    "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", "berdatangan",
    "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali-kali", "berkata", "berkehendak",
    "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan", "bermacam",
    "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama", "bersiap", "bersiap-siap",
    "bertanya", "bertanya-tanya", "berturut", "berturut-turut", "bertutur", "berujar", "berupa", "besar",
    "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "boleh", "bolehkah",
    "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya",
    "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang",
    "dekat", "demi", "demikian", "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya",
    "dialah", "diantara", "diantaranya", "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya",
    "didapat", "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", "diingat", "diingatkan",
    "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya",
    "dikerjakan", "diketahui", "diketahuinya", "dikiranya", "dilakukan", "dilalui", "dilihat", "dimaksud",
    "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai", "dimisalkan", "dimulai",
    "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", "diperbuatnya",
    "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya", "dipersoalkan",
    "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan", "disebut", "disebutkan", "disebutkannya",
    "disini", "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan",
    "ditujukan", "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "dituturkan", "dituturkannya",
    "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya",
    "entah", "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus",
    "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibu", "ikut",
    "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah",
    "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban",
    "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya",
    "justru", "kala", "kalau", "kalaulah", "kalaupun", "kali", "kalian", "kami", "kamilah", "kamu",
    "kamulah", "kan", "kapan", "kapankah", "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan",
    "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya", "keinginan",
    "kelak", "kelima", "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada",
    "kepadanya", "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya",
 "atas", "untuk", "pada", "yg", "ga", "gak", "gk", "engga", "nggak", "nya", "sih", "aja", "saja", "deh", "kok",
    "klo", "kalo", "biar", "udah", "sudah", "tp", "tapi", "sy", "saya", "aku", "gua", "gue"
]

# ==============================================================================
# 1. Muat Model AI (Hanya sekali saat aplikasi dimulai)
# ==============================================================================
print("⏳ Memuat model sentiment analysis... Ini hanya dilakukan sekali saat startup.")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="crypter70/IndoBERT-Sentiment-Analysis",
        tokenizer="crypter70/IndoBERT-Sentiment-Analysis"
    )
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model. Error: {e}")
    raise SystemExit("Eksekusi dihentikan karena model tidak dapat dimuat.")

# ==============================================================================
# 2. Definisikan Fungsi-Fungsi Inti
# ==============================================================================
def get_playstore_reviews_dataframe(app_id: str, count: int = 100, lang: str = 'id', country: str = 'id'):
    """Mengambil ulasan dari Google Play Store dan mengembalikan DataFrame."""
    print(f"⏳ Mengambil {count} ulasan untuk {app_id}...")
    all_reviews = []
    continuation_token = None
    while len(all_reviews) < count:
        try:
            result, token = reviews(
                app_id, lang=lang, country=country, sort=Sort.NEWEST,
                count=min(count - len(all_reviews), 200),
                continuation_token=continuation_token
            )
            if not result: break
            all_reviews.extend(result)
            continuation_token = token
            if not continuation_token: break
        except Exception as e:
            print(f"⚠️ Error saat scraping: {e}")
            break
    if not all_reviews:
        return None
    print(f"✅ Berhasil mengambil {len(all_reviews[:count])} ulasan.")
    return pd.DataFrame(all_reviews[:count])

def clean_text(text: str) -> str:
    """Membersihkan teks ulasan."""
    if not isinstance(text, str): return ""
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text) # Hapus angka
    return text.strip().lower()

def analyze_sentiment(text: str) -> str:
    """Menganalisis sentimen dari teks yang sudah bersih."""
    if not text or not text.strip(): return "NEUTRAL"
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)
        return result[0]['label']
    except Exception:
        return "NEUTRAL"

# ==============================================================================
# 2.1. Fungsi Baru untuk Visualisasi
# ==============================================================================
def create_image_base64(figure):
    """Mengubah figure matplotlib menjadi string base64."""
    buf = BytesIO()
    figure.savefig(buf, format="png", bbox_inches='tight')
    plt.close(figure) # Tutup figure untuk membebaskan memori
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_wordcloud(text_corpus: str):
    """Membuat WordCloud dan mengembalikannya sebagai base64."""
    if not text_corpus.strip(): return None
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=STOPWORDS_ID, collocations=False
    ).generate(text_corpus)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return create_image_base64(fig)

def generate_top_words_plot(text_corpus: str, top_n: int = 10):
    """Membuat plot bar untuk kata paling umum dan mengembalikannya sebagai base64."""
    if not text_corpus.strip(): return None
    words = [word for word in text_corpus.split() if word not in STOPWORDS_ID]
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(top_n)

    if not most_common_words: return None

    df_top_words = pd.DataFrame(most_common_words, columns=['word', 'count']).sort_values(by='count')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_top_words['word'], df_top_words['count'], color='skyblue')
    ax.set_title(f'Top {top_n} Kata yang Sering Muncul')
    ax.set_xlabel('Frekuensi')
    plt.tight_layout()
    return create_image_base64(fig)

# ==============================================================================
# 3. Bangun Aplikasi FastAPI
# ==============================================================================
app = FastAPI(
    title="API Analisis Sentimen Ulasan Google Play",
    description="API untuk mengambil ulasan aplikasi, membersihkan, menganalisis sentimen, dan membuat visualisasi (WordCloud & Top Words).",
    version="1.1.0"
)

class ReviewRequest(BaseModel):
    app_id: str
    count: int = 100

@app.post("/analyze_reviews")
async def analyze_reviews_endpoint(request: ReviewRequest):
    """Endpoint untuk menjalankan pipeline analisis sentimen lengkap."""
    df_raw = get_playstore_reviews_dataframe(request.app_id, count=request.count)
    if df_raw is None or df_raw.empty:
        raise HTTPException(status_code=404, detail=f"Tidak ada ulasan yang ditemukan untuk app_id: {request.app_id}")

    df = df_raw[['content']].copy()
    df.rename(columns={'content': 'original_review'}, inplace=True)

    print("🚀 Menjalankan pipeline analisis...")
    df['cleaned_review'] = df['original_review'].apply(clean_text)
    df['sentiment'] = df['cleaned_review'].apply(analyze_sentiment)
    print("✅ Pipeline analisis selesai.")

    # Hitung distribusi sentimen dasar
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    
    # Siapkan struktur data baru untuk hasil akhir
    sentiment_analysis_results = {}

    print("📊 Membuat visualisasi untuk setiap sentimen...")
    # Loop melalui setiap sentimen yang ditemukan (Positive, Negative, Neutral)
    for sentiment_label, count in sentiment_counts.items():
        # Gabungkan semua teks dari ulasan dengan sentimen yang sama
        text_corpus = ' '.join(df[df['sentiment'] == sentiment_label]['cleaned_review'])

        # Buat visualisasi
        wordcloud_image = generate_wordcloud(text_corpus)
        top_words_plot = generate_top_words_plot(text_corpus, top_n=10)

        # Simpan hasilnya
        sentiment_analysis_results[sentiment_label] = {
            "count": count,
            "wordcloud_image_base64": wordcloud_image,
            "top_words_plot_base64": top_words_plot
        }
    print("✅ Visualisasi selesai.")

    return {
        "app_id": request.app_id,
        "review_count": len(df),
        "sentiment_analysis": sentiment_analysis_results,
        "reviews": df.to_dict('records')
    }

@app.get("/")
async def read_root():
    return {"message": "Selamat datang! API Analisis Sentimen aktif. Buka /docs untuk mencoba."}
