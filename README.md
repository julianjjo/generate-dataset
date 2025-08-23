# Dataset Generator con Ollama

Generador de datasets masivos para entrenamiento de modelos de lenguaje usando Ollama. Capaz de generar hasta 100 millones de ejemplos con diferentes tipos de contenido de alta calidad.

## 🚀 Características

- **Generación masiva**: Soporte para datasets de hasta 100M de ejemplos
- **Soporte multiidioma**: Español, inglés o contenido mixto
- **Múltiples tipos de contenido**: Cuentos, instrucciones, código, artículos, diálogos y ensayos
- **Formato optimizado**: Compatible con `tokenize_function` estándar
- **Procesamiento asíncrono**: Generación eficiente con control de concurrencia
- **Sistema de checkpoints**: Recuperación automática en caso de interrupciones
- **Progreso en tiempo real**: Logs detallados y barra de progreso actualizada
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
# Generar 1000 ejemplos en español (por defecto)
python main.py --size 1000

# Generar dataset pequeño en inglés
python main.py --size 100 --batch-size 10 --language en --output english_dataset

# Generar dataset mixto (español + inglés)
python main.py --size 500 --language mixed --output multilingual_dataset
```

### Configuración Avanzada
```bash
# Dataset masivo en inglés con modelo específico
python main.py --size 10000000 --model codellama --batch-size 200 --concurrent 30 --language en

# Dataset mixto usando servidor Ollama remoto
python main.py --ollama-url http://192.168.1.100:11434 --model llama3.1 --size 50000 --language mixed

# Dataset especializado en código con CodeLlama
python main.py --model codellama --size 25000 --language en --output code_dataset
```

### Solo Consolidación
```bash
# Consolidar archivos existentes sin generar nuevos
python main.py --consolidate-only --output mi_dataset
```

## 🌍 Soporte Multiidioma

El generador soporta tres modos de idioma:

| Modo | Descripción | Uso |
|------|-------------|-----|
| **Español (`es`)** | Todo el contenido en español | `--language es` (por defecto) |
| **Inglés (`en`)** | Todo el contenido en inglés | `--language en` |
| **Mixto (`mixed`)** | Alterna aleatoriamente entre español e inglés | `--language mixed` |

### Ejemplos de Uso por Idioma

```bash
# Dataset completamente en español
python main.py --size 10000 --language es --output spanish_dataset

# Dataset completamente en inglés  
python main.py --size 10000 --language en --output english_dataset

# Dataset mixto (ideal para modelos multilingües)
python main.py --size 10000 --language mixed --output multilingual_dataset
```

## 📊 Tipos de Dataset Generados

El generador crea 6 tipos diferentes de contenido en ambos idiomas:

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
| `--language` | Idioma del dataset | "es" |
| `--consolidate-only` | Solo consolidar archivos existentes | False |

### Opciones de Idioma

| Valor | Descripción |
|-------|-------------|
| `es` | Genera todo el contenido en español |
| `en` | Genera todo el contenido en inglés |
| `mixed` | Alterna aleatoriamente entre español e inglés por ejemplo |

### Ejemplos de Uso por Escenario

#### Dataset para Fine-tuning General en Español
```bash
python main.py --size 100000 --model llama3.1 --batch-size 50 --language es --output spanish_general
```

#### Dataset de Código en Inglés
```bash
python main.py --size 50000 --model codellama --batch-size 25 --language en --output english_code
```

#### Dataset Multilingüe Masivo (Producción)
```bash
python main.py --size 50000000 --batch-size 500 --concurrent 50 --language mixed --output multilingual_production
```

#### Dataset Especializado por Idioma
```bash
# Instrucciones técnicas en inglés
python main.py --size 25000 --model llama3.1 --language en --output tech_instructions_en

# Contenido creativo en español
python main.py --size 25000 --model llama3.1 --language es --output creative_content_es
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

## 🔄 Sistema de Checkpoints y Progreso

El generador incluye un sistema robusto de checkpoints y monitoreo en tiempo real:

### Checkpoints Automáticos
- **Guardado automático**: Cada 10,000 ejemplos generados
- **Recuperación automática**: Reanuda desde el último checkpoint
- **Información de progreso**: Tracking detallado del avance

### Monitoreo en Tiempo Real
- **Logs detallados**: Información de cada lote procesado
- **Barra de progreso**: Actualización visual continua
- **Contadores dinámicos**: Ejemplos generados y porcentaje completado
- **Indicadores visuales**: Emojis para fácil identificación (✓, 💾)

### Ejemplo de Salida de Progreso
```
2025-08-23 23:18:19,489 - INFO - Iniciando generación de dataset: 10,000 ejemplos
2025-08-23 23:18:21,279 - INFO - Conexión con Ollama establecida
2025-08-23 23:18:22,156 - INFO - Procesando lote 1/100
2025-08-23 23:18:25,789 - INFO - ✓ Guardado lote 1: 100 elementos | Total: 100
2025-08-23 23:18:26,234 - INFO - Lote 1 completado: 100 ejemplos generados

Generando dataset: 15%|████████████                     | 15/100 lotes [ejemplos: 1,500, progreso: 15.0%]

2025-08-23 23:25:34,123 - INFO - 💾 Checkpoint guardado: 10,000 elementos (100.0%)
```

### Formato de Checkpoint
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

#### Problemas con idiomas específicos
- **Contenido en idioma incorrecto**: Verificar el parámetro `--language`
- **Mezcla inconsistente**: En modo `mixed`, la alternancia es aleatoria por diseño
- **Modelos especializados**: Algunos modelos funcionan mejor con idiomas específicos:
  - `llama3.1`: Excelente para español e inglés
  - `codellama`: Mejor para código en inglés
  - `mistral`: Bueno para contenido multilingüe

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

## 💡 Tips y Mejores Prácticas

### Para Datasets Multilingües
- **Modo mixto**: Ideal para entrenar modelos que necesiten responder en ambos idiomas
- **Datasets separados**: Para fine-tuning específico por idioma, genera datasets individuales
- **Verificación de calidad**: Revisa algunos ejemplos para asegurar la calidad del idioma

### Para Datasets Masivos
- **Servidores dedicados**: Para datasets muy grandes, usa un servidor con buena conectividad
- **Monitoreo continuo**: Las mejoras de progreso te permiten monitorear generaciones largas
- **Checkpoints**: Los checkpoints automáticos permiten reanudar generaciones interrumpidas

### Para Rendimiento Óptimo
- **Concurrencia balanceada**: Más concurrent tasks = mayor memoria, pero también mayor velocidad
- **Batch size apropiado**: Lotes más grandes son más eficientes pero consumen más memoria
- **Modelo adecuado**: Elige el modelo según el tipo de contenido que necesites

⚡ **Recomendación**: Para datasets de producción, inicia con una prueba pequeña usando `--size 1000` para verificar calidad y rendimiento antes de generar el dataset completo.
