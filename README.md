# Dataset Generator con Ollama

Generador de datasets masivos para entrenamiento de modelos de lenguaje usando Ollama. Capaz de generar hasta 100 millones de ejemplos con diferentes tipos de contenido de alta calidad.

## 🚀 Características

- **Generación masiva**: Soporte para datasets de hasta 100M de ejemplos
- **Múltiples tipos de contenido**: Cuentos, instrucciones, código, artículos, diálogos y ensayos
- **Formato optimizado**: Compatible con `tokenize_function` estándar
- **Procesamiento asíncrono**: Generación eficiente con control de concurrencia
- **Sistema de checkpoints**: Recuperación automática en caso de interrupciones
- **Consolidación automática**: Combina múltiples archivos en un dataset final

## 📋 Requisitos

- Python 3.8+
- Ollama instalado y ejecutándose
- Al menos un modelo de Ollama descargado (ej: `llama3.1`, `codellama`)

## 🛠️ Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repository-url>
   cd generate-dataset
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Instalar y configurar Ollama**:
   ```bash
   # Descargar Ollama desde https://ollama.ai
   
   # Descargar un modelo (ejemplo)
   ollama pull llama3.1
   ollama pull codellama
   ```

4. **Verificar instalación**:
   ```bash
   ollama list  # Debe mostrar los modelos descargados
   ollama serve # Iniciar servidor (puerto 11434 por defecto)
   ```

## 🎯 Uso Básico

### Generación Simple
```bash
# Generar 1000 ejemplos con configuración por defecto
python main.py --size 1000

# Generar dataset pequeño para pruebas
python main.py --size 100 --batch-size 10 --output prueba_dataset
```

### Configuración Avanzada
```bash
# Dataset masivo con modelo específico
python main.py --size 10000000 --model codellama --batch-size 200 --concurrent 30

# Usar servidor Ollama remoto
python main.py --ollama-url http://192.168.1.100:11434 --model llama3.1 --size 50000
```

### Solo Consolidación
```bash
# Consolidar archivos existentes sin generar nuevos
python main.py --consolidate-only --output mi_dataset
```

## 📊 Tipos de Dataset Generados

El generador crea 6 tipos diferentes de contenido:

| Tipo | Descripción | Tamaño típico |
|------|-------------|---------------|
| **Cuentos** | Narrativas completas con inicio, desarrollo y final | 300-500 palabras |
| **Instrucciones** | Guías paso a paso educativas y técnicas | 200+ palabras |
| **Diálogos** | Conversaciones naturales con contexto | 8-10 intercambios |
| **Artículos** | Textos informativos estructurados | 400-600 palabras |
| **Código** | Programas completos con comentarios | Funcional y documentado |
| **Ensayos** | Textos reflexivos y académicos | 400-500 palabras |

## 🔧 Parámetros de Configuración

### Argumentos de línea de comandos

```bash
python main.py [opciones]
```

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--size` | Número total de ejemplos a generar | 100,000,000 |
| `--batch-size` | Ejemplos por lote | 100 |
| `--concurrent` | Tareas concurrentes máximas | 20 |
| `--output` | Directorio de salida | "generated_dataset" |
| `--ollama-url` | URL del servidor Ollama | "http://localhost:11434" |
| `--model` | Modelo de Ollama a usar | "llama3.1" |
| `--consolidate-only` | Solo consolidar archivos existentes | False |

### Ejemplos de Uso por Escenario

#### Dataset para Fine-tuning General
```bash
python main.py --size 100000 --model llama3.1 --batch-size 50 --output general_dataset
```

#### Dataset de Código
```bash
python main.py --size 50000 --model codellama --batch-size 25 --output code_dataset
```

#### Dataset Masivo (Producción)
```bash
python main.py --size 50000000 --batch-size 500 --concurrent 50 --output production_dataset
```

## 📁 Estructura de Salida

```
mi_dataset/
├── batch_000001.jsonl    # Lotes individuales
├── batch_000002.jsonl
├── ...
├── checkpoint.json       # Progreso guardado
└── complete_dataset.jsonl # Dataset consolidado (opcional)
```

### Formato de Datos

Cada línea en los archivos `.jsonl` tiene el formato:

```json
{"text": "Contenido completo del ejemplo aquí..."}
```

Este formato es **directamente compatible** con la función `tokenize_function` estándar que busca el campo `text`.

## 🔄 Sistema de Checkpoints

El generador incluye un sistema robusto de checkpoints:

- **Guardado automático**: Cada 10,000 ejemplos generados
- **Recuperación automática**: Reanuda desde el último checkpoint
- **Información de progreso**: Tracking detallado del avance

Ejemplo de checkpoint:
```json
{
  "generated_count": 50000,
  "timestamp": 1692123456.789,
  "progress": 50.0
}
```

## 📈 Rendimiento y Optimización

### Recomendaciones por Tamaño de Dataset

| Tamaño del Dataset | Batch Size | Concurrent | Tiempo Estimado* |
|-------------------|------------|------------|------------------|
| 1K - 10K | 10-25 | 5-10 | 10-30 min |
| 10K - 100K | 25-100 | 10-20 | 1-5 horas |
| 100K - 1M | 100-200 | 20-30 | 5-20 horas |
| 1M+ | 200-500 | 30-50 | 20+ horas |

*Los tiempos dependen del modelo, hardware y configuración de Ollama.

### Consejos de Optimización

1. **Ajustar concurrencia**: Más concurrent tasks = mayor uso de memoria
2. **Batch size óptimo**: Balance entre memoria y eficiencia de red
3. **Modelo adecuado**: Modelos más pequeños = generación más rápida
4. **Recursos del sistema**: Monitor CPU y memoria durante generación

## 🐛 Solución de Problemas

### Errores Comunes

#### "Cannot connect to host localhost:11434"
```bash
# Verificar que Ollama esté ejecutándose
ollama serve

# En otra terminal, probar conexión
curl http://localhost:11434/api/tags
```

#### "Model not found"
```bash
# Listar modelos disponibles
ollama list

# Descargar el modelo necesario
ollama pull llama3.1
```

#### "Out of memory"
- Reducir `--concurrent` y `--batch-size`
- Usar un modelo más pequeño
- Cerrar otras aplicaciones que consuman memoria

#### Generación muy lenta
- Verificar recursos del sistema (CPU, memoria)
- Usar un modelo más rápido (ej: `llama3.1` vs `llama3.1:70b`)
- Ajustar parámetros de concurrencia

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Add nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🔗 Enlaces Útiles

- [Ollama Official Website](https://ollama.ai)
- [Ollama Models Library](https://ollama.ai/library)
- [Python AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)

---

⚡ **Tip**: Para datasets muy grandes, considera ejecutar el generador en un servidor dedicado con buena conectividad y recursos computacionales adecuados.
