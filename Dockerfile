FROM python:3.10-slim

# Evita interações durante o apt install
ENV DEBIAN_FRONTEND=noninteractive

# Instala libGL + dependências básicas para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /code

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Comando padrão
CMD ["python", "src/main.py"]
