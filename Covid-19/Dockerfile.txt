# Imagen base
FROM python:3.10

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY main.py /app
COPY modelo_covid19.keras /app

# Instalar dependencias
RUN pip install flask tensorflow pillow
RUN pip install flask-cors

# Exponer puerto
EXPOSE 8000

# Comando para iniciar la API
CMD ["python", "main.py"]
