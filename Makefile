# Makefile para Bike Sharing Prediction API
# Variables configurables

# Registry y nombre de la imagen (formato: registry/imagen)
IMAGE_NAME ?= jsdevart/mlops

# Tag por defecto (puede ser sobrescrito: make build TAG=v1.0.0)
TAG ?= latest

# Construir el nombre completo de la imagen
FULL_IMAGE_NAME = $(IMAGE_NAME):$(TAG)

# Dockerfile a usar
DOCKERFILE ?= Dockerfile

# Puerto por defecto
PORT ?= 8000

.PHONY: build push run

build: ## Construir la imagen Docker
	@echo "🔨 Construyendo imagen: $(FULL_IMAGE_NAME)"
	docker build -t $(FULL_IMAGE_NAME) -f $(DOCKERFILE) .
	@echo "✓ Imagen construida: $(FULL_IMAGE_NAME)"

push: ## Hacer push de la imagen al registry
	@echo "📤 Haciendo push de: $(FULL_IMAGE_NAME)"
	docker push $(FULL_IMAGE_NAME)
	@echo "✓ Push completado: $(FULL_IMAGE_NAME)"

run: ## Ejecutar el contenedor en modo interactivo
	@echo "🚀 Ejecutando contenedor: $(FULL_IMAGE_NAME)"
	docker run -it --rm -p $(PORT):8000 $(FULL_IMAGE_NAME)
