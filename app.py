import os
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2
import cv2.data

app = Flask(__name__)

# Yüklenen ve işlenen görsellerin tutulacağı klasör
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# YOLO modelini yükle (extra large)
model = YOLO("yolov8l.pt")

# OpenCV'nin hazır yüz algılayıcısı (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Kategori -> YOLO sınıf etiketleri eşlemesi
CATEGORY_MAP = {
    "human": ["person"],
    "animal": ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["car", "bus", "truck", "train", "motorbike", "bicycle"]
}


def analyze_image(image_path: str):
    """
    YOLO ile nesne tespiti yapar,
    bounding box çizilmiş resmi kaydeder,
    nesne etiketlerini ve güven skorlarını döner.
    """
    results = model(image_path)[0]  # tek resim

    # YOLO'nun çizdiği görüntü (numpy array, BGR)
    annotated_img = results.plot()

    # Yeni dosya adı: orijinal_ismi_pred.uzanti
    base_name = os.path.basename(image_path)      # ornek: resim.jpg
    name, ext = os.path.splitext(base_name)       # name=resim, ext=.jpg
    annotated_name = f"{name}_pred{ext}"          # resim_pred.jpg
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)

    # Annotated görüntüyü kaydet
    cv2.imwrite(annotated_path, annotated_img)

    # Tespit edilen nesneleri listele
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]         # sınıf adı (person, dog vs.)
        conf = float(box.conf[0]) * 100.0     # yüzde cinsinden

        detections.append({
            "label": label,
            "conf": round(conf, 1)
        })

    return annotated_name, detections


def detect_faces(image_path: str):
    """
    Resimdeki yüzleri tespit eder.
    Her yüz için (x, y, w, h) box bilgisi döndürür.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scaleFactor ve minNeighbors: hassasiyet ayarları
    faces_raw = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    faces = []
    for (x, y, w, h) in faces_raw:
        faces.append({
            "box": [int(x), int(y), int(w), int(h)]
        })

    return faces


@app.route("/", methods=["GET", "POST"])
def index():
    original_url = None   # orijinal resim
    pred_url = None       # YOLO çıktısı
    detections = []       # (filtrelenmiş) tespit edilen nesneler
    faces = []            # tespit edilen yüzler
    selected_categories = []  # seçilen filtreler (human/animal/vehicle)
    all_detections_count = 0  # YOLO'nun bulduğu toplam nesne sayısı

    if request.method == "POST":
        if "image" not in request.files:
            return "Herhangi bir dosya gönderilmedi.", 400

        file = request.files["image"]
        if file.filename == "":
            return "Dosya seçmedin.", 400

        # Dosyayı kaydet
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        # Orijinal görüntünün URL'i
        original_url = url_for("static", filename=f"uploads/{file.filename}")

        # YOLO analizi (tüm nesneler)
        annotated_name, all_detections = analyze_image(save_path)
        pred_url = url_for("static", filename=f"uploads/{annotated_name}")
        all_detections_count = len(all_detections)

        # Formdan seçilen kategori filtrelerini al
        selected_categories = request.form.getlist("category")  # ["human", "animal"] gibi

        # Eğer kategori seçilmemişse: tüm nesneleri göster
        if selected_categories:
            allowed_labels = set()
            for cat in selected_categories:
                allowed_labels.update(CATEGORY_MAP.get(cat, []))

            detections = [
                d for d in all_detections
                if d["label"] in allowed_labels
            ]
        else:
            # Filtre seçilmediyse, hepsini göster
            detections = all_detections

        # Yüz tespiti
        faces = detect_faces(save_path)

    # HEM GET HEM POST durumunda burası çalışır
    return render_template(
        "index.html",
        original_url=original_url,
        pred_url=pred_url,
        detections=detections,
        faces=faces,
        selected_categories=selected_categories,
        all_detections_count=all_detections_count
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



