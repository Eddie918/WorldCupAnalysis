from PIL import Image

# Cargar la imagen original
image_path = "soccerfield.jpg"  # Reemplaza con la ruta a tu imagen
image = Image.open(image_path)

# Redimensionar la imagen a 612x612 píxeles
image_square = image.resize((612, 612))

# Guardar la imagen ajustada
adjusted_image_path = "soccer_field_adjusted.png"  # Reemplaza con la ruta donde quieres guardar la imagen ajustada
image_square.save(adjusted_image_path)

print(f"Imagen ajustada guardada en: {adjusted_image_path}")
