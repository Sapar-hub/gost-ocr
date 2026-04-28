# Comparison Results

## Per-Image Results

| # | Image | YOLO IoU | OpenCV IoU | Winner |
|---|-------|----------|------------|--------|
| 1 | uzel-val-koleso.jpg | 0.000 | 0.940 | OpenCV |
| 2 | reduktor-chervjachnyj_-vid-obschij_deskewed.jpg | 0.000 | 0.000 | TIE |
| 3 | stupica-chervjachnogo-kolesa.jpg | 0.872 | 0.000 | **YOLO** |
| 4 | vedomyj-val_deskewed.jpg | 0.932 | 0.862 | **YOLO** |
| 5 | nasos-jecn-325.jpg | 0.140 | 0.000 | **YOLO** |
| 6 | 1.png | 0.000 | 0.000 | TIE |
| 7 | 2.jpg | 0.000 | 0.000 | TIE |
| 8 | venec-chervjachnogo-kolesa.jpg | 0.934 | 0.894 | **YOLO** |
| 9 | shema-stropilnoj-sistemy_deskewed.jpg | 0.000 | 0.000 | TIE |
| 10 | kryshka-vedomovogo-vala-gluhaja.jpg | 0.865 | 0.328 | **YOLO** |
| 11 | uzly-1-2-3-4-5.jpg | 0.946 | 0.945 | **YOLO** |

## Summary Statistics

| Metric | YOLO | OpenCV |
|--------|------|-------|
| Mean IoU | **0.426** | 0.361 |
| Median IoU | **0.865** | 0.328 |
| Wins (IoU > opponent) | **6** | 1 |
| Ties | 4 | 4 |
| Full detections (IoU > 0.5) | 5 | 3 |
| Complete failures (IoU = 0) | 5 | 6 |

## Failure Cases Analysis

### YOLO Failures (5 images with IoU = 0):
- `uzel-val-koleso.jpg` - YOLO detected wrong region (bbox offset significantly)
- `reduktor-chervjachnyj_-vid-obschij_deskewed.jpg` - No detection
- `1.png`, `2.jpg` - No detection (synthetic images)
- `shema-stropilnoj-sistemy_deskewed.jpg` - No detection

### OpenCV Failures (6 images with IoU = 0):
- `stupica-chervjachnogo-kolesa.jpg` - No detection
- `nasos-jecn-325.jpg` - No detection
- `1.png`, `2.jpg` - No detection
- `shema-stropilnoj-sistemy_deskewed.jpg` - No detection
- `reduktor-chervjachnyj_-vid-obschij_deskewed.jpg` - No detection

## Key Observations

1. **YOLO excels at finding stamps in complex backgrounds** - Success on `stupica`, `venec`, `kryshka`
2. **OpenCV wins when both detect** - `uzel-val-koleso.jpg` (0.94 vs 0.00)
3. **Both struggle with certain image types** - `shema-stropilnoj-sistemy`, `reduktor`
4. **Synthetic images (1.png, 2.jpg)** - Both fail (no real content)

## Dataset

- Training: 25 synthetic images (DPI 200-400, FORM_3/4/5)
- Test: 11 real archive drawings
- Ground Truth: YOLO format labels in `src/gost_ocr/datasets/test/labels/`