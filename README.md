## Cek Melon

Base URL :

> https://melon-8080.apps.cs.ipb.ac.id


## `Upload Image`

### POST /upload

### https://melon-8080.apps.cs.ipb.ac.id/upload

### Description
####  Mengunggah gambar melon untuk dilakukan prediksi kematangan.
### Request body

| Fieldname | Type     | Necessity    | Desc |
| --------- | -------- | ------------ | ---- |
| file      | `file`   | **required** |      |
| longitude | `float`  | **required** |      |
| latitude  | `float`  | **required** |      |


### Sample Successful Responses

```json
{
  "message": "Image uploaded successfully"
}
```

## `Predict`

### GET /predict

### https://melon-8080.apps.cs.ipb.ac.id/predict


### Sample success response

```json
{
  "GLCM": {
    "Before normalized": {
      "contrast": 0.0,
      "energy": 0.0,
      "homogeneity": 0.0,
      "correlation": 0.0,
      "dissimilarity": 0.0,
      "jumlah piksel jala": 0,
      "kepadatan piksel jala": 0.0
    },
    "Normalized": {
      "contrast": 0.0,
      "energy": 0.0,
      "homogeneity": 0.0,
      "correlation": 0.0,
      "dissimilarity": 0.0,
      "jumlah piksel jala": 0,
      "kepadatan piksel jala": 0.0
    }
  },
  "Prediction_SVM": "Belum matang",
  "Prediction_RF": "Siap panen",
  "image_filename": "image.jpg"
}

```
