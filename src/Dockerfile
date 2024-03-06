# Usa una imagen base de Python
FROM python:3.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install -r requirements.txt

# Copia el resto del código al contenedor
COPY . .

# Comando para ejecutar tu aplicación (debes reemplazar 'app.py' con el nombre de tu archivo principal de la aplicación)
CMD ["python", "main.py"]
