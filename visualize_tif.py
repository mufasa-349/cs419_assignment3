#!/usr/bin/env python3
"""
TIF label görüntüsünü görselleştir
"""
import cv2
import numpy as np
import sys
from pathlib import Path

def visualize_label_tif(tif_path: str, output_path: str = None):
    """TIF label dosyasını renkli görselleştirmeye çevir"""
    # TIF dosyasını oku (16-bit veya 8-bit)
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Hata: {tif_path} dosyası okunamadı!")
        return
    
    print(f"Dosya: {tif_path}")
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    print(f"Min: {img.min()}, Max: {img.max()}")
    print(f"Unique labels: {len(np.unique(img))}")
    
    # Label görüntüsünü renkli hale getir
    # Her label'e farklı bir renk atayalım
    labels = np.unique(img)
    n_labels = len(labels) - 1  # 0'ı saymıyoruz (background)
    
    # RGB görüntü oluştur
    h, w = img.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Her label için rastgele renk oluştur (deterministic için seed)
    np.random.seed(42)
    for label in labels:
        if label == 0:
            # Background siyah
            colored[img == label] = [0, 0, 0]
        else:
            # Her label için parlak renk
            color = np.random.randint(50, 255, size=3, dtype=np.uint8)
            colored[img == label] = color
    
    # Ayrıca normalize edilmiş grayscale versiyonu da oluştur
    if img.max() > 0:
        gray_norm = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
        gray_colored = cv2.applyColorMap(gray_norm, cv2.COLORMAP_JET)
    else:
        gray_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Çıktı dosyası adı
    if output_path is None:
        tif_path_obj = Path(tif_path)
        output_path = str(tif_path_obj.with_name(tif_path_obj.stem + "_visualized.png"))
    
    # İki görüntüyü yan yana kaydet
    combined = np.hstack([colored, gray_colored])
    cv2.imwrite(output_path, combined)
    print(f"Görselleştirme kaydedildi: {output_path}")
    print(f"  Sol: Renkli label görüntüsü (her label farklı renk)")
    print(f"  Sağ: Normalize edilmiş grayscale (JET colormap)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python visualize_tif.py <tif_dosyası> [çıktı_dosyası]")
        sys.exit(1)
    
    tif_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_label_tif(tif_path, output_path)

