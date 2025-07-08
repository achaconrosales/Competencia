# 🔬 Segmentación de Esporas con U-Net

Este proyecto implementa una red neuronal **U-Net** en PyTorch para la segmentación semántica de esporas en imágenes.
![Ejemplo de segmentación](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)

El objetivo es clasificar cada píxel de una imagen como perteneciente a una "espora" o al "fondo", permitiendo identificar visualmente la presencia y extensión de las esporas. A continuación, se muestra un ejemplo de imagen original junto a su máscara binaria generada:

| Imagen original | Máscara generada manualmente| Máscara predicha con 25 imagenes de entrenamiento
|------------------|------------------|------------------|
| ![Espora](dataset/images/ectomicorrizas1.png) | ![Mascara](dataset/masks/ectomicorrizas1.png) | ![Predicción](predictions/combined_ectomicorrizas1.png)
