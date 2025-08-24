# Massive Dataset Generator with Ollama

Massive dataset generator for training language models using Ollama. Capable of generating up to 100 million high-quality examples with different types of content.

## 🚀 Features

- **Massive generation**: Support for datasets up to 100M examples
- **Multilingual support**: English, Spanish, or mixed content
- **Multiple content types**: Stories, instructions, code, articles, dialogues, and essays
- **Optimized format**: Compatible with standard `tokenize_function`
- **Asynchronous processing**: Efficient generation with concurrency control
- **Checkpoint system**: Automatic recovery in case of interruptions
- **Real-time progress**: Detailed logs and updated progress bar
- **Automatic consolidation**: Combines multiple files into a final dataset
- **Custom timeouts**: Configurable generation timeouts for different models

## 📋 Requirements

- Python 3.8+
- Ollama installed and running
- At least one Ollama model downloaded (e.g., `llama3.1`, `codellama`, `nemotron`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd generate-dataset
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama**:
   ```bash
   # Download Ollama from https://ollama.ai
   
   # Download a model (example)
   ollama pull llama3.1
   ollama pull codellama
   ollama pull nemotron:latest
   ```

4. **Verify installation**:
   ```bash
   ollama list  # Should show downloaded models
   ollama serve # Start server (default port 11434)
   ```

## 🎯 Basic Usage

### Simple Generation
```bash
# Generate 1000 examples in English (default changed to English)
python main.py --size 1000 --language en

# Generate small dataset in Spanish
python main.py --size 100 --batch-size 10 --language es --output spanish_dataset

# Generate mixed dataset (Spanish + English)
python main.py --size 500 --language mixed --output multilingual_dataset
```

### Advanced Configuration
```bash
# Massive English dataset with specific model
python main.py --size 10000000 --model codellama --batch-size 200 --concurrent 30 --language en

# Mixed dataset using remote Ollama server
python main.py --ollama-url http://192.168.1.100:11434 --model llama3.1 --size 50000 --language mixed

# Specialized code dataset with CodeLlama
python main.py --model codellama --size 25000 --language en --output code_dataset

# High-performance generation with custom timeout
python main.py --size 10000000 --model nemotron:latest --batch-size 300 --concurrent 15 --language en --timeout 1200
```

### Consolidation Only
```bash
# Consolidate existing files without generating new ones
python main.py --consolidate-only --output my_dataset
```

## 🌍 Multilingual Support

The generator supports three language modes:

| Mode | Description | Usage |
|------|-------------|-------|
| **English (`en`)** | All content in English | `--language en` |
| **Spanish (`es`)** | All content in Spanish | `--language es` |
| **Mixed (`mixed`)** | Randomly alternates between Spanish and English | `--language mixed` |

### Usage Examples by Language

```bash
# Dataset completely in English
python main.py --size 10000 --language en --output english_dataset

# Dataset completely in Spanish  
python main.py --size 10000 --language es --output spanish_dataset

# Mixed dataset (ideal for multilingual models)
python main.py --size 10000 --language mixed --output multilingual_dataset
```

## 📊 Generated Dataset Types

The generator creates 6 different types of content in both languages:

| Type | Description | Typical Size |
|------|-------------|--------------|
| **Stories** | Complete narratives with beginning, development, and ending | 300-500 words |
| **Instructions** | Step-by-step educational and technical guides | 200+ words |
| **Dialogues** | Natural conversations with context | 8-10 exchanges |
| **Articles** | Structured informative texts | 400-600 words |
| **Code** | Complete programs with comments | Functional and documented |
| **Essays** | Reflective and academic texts | 400-500 words |

## 🔧 Configuration Parameters

### Command Line Arguments

```bash
python main.py [options]
```

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--size` | Total number of examples to generate | 100,000,000 |
| `--batch-size` | Examples per batch | 100 |
| `--concurrent` | Maximum concurrent tasks | 20 |
| `--output` | Output directory | "generated_dataset" |
| `--ollama-url` | Ollama server URL | "http://localhost:11434" |
| `--model` | Ollama model to use | "llama3.1" |
| `--language` | Dataset language | "es" |
| `--timeout` | Custom timeout in seconds | Automatic based on model |
| `--consolidate-only` | Only consolidate existing files | False |
| `--cpu-tips` | Show CPU optimization tips | False |

### Language Options

| Value | Description |
|-------|-------------|
| `en` | Generates all content in English |
| `es` | Generates all content in Spanish |
| `mixed` | Randomly alternates between Spanish and English per example |

### Usage Examples by Scenario

#### General Fine-tuning Dataset in English
```bash
python main.py --size 100000 --model llama3.1 --batch-size 50 --language en --output english_general
```

#### Code Dataset in English
```bash
python main.py --size 50000 --model codellama --batch-size 25 --language en --output english_code
```

#### Massive Multilingual Dataset (Production)
```bash
python main.py --size 50000000 --batch-size 500 --concurrent 50 --language mixed --output multilingual_production --timeout 600
```

#### Specialized Dataset by Language
```bash
# Technical instructions in English
python main.py --size 25000 --model llama3.1 --language en --output tech_instructions_en

# Creative content in Spanish
python main.py --size 25000 --model llama3.1 --language es --output creative_content_es
```

#### CPU Optimization for Large Models
```bash
# Large model on CPU with optimized settings
python main.py --size 1000000 --model nemotron:latest --batch-size 50 --concurrent 5 --language en --timeout 1800 --cpu-tips
```

## 📁 Output Structure

```
my_dataset/
├── batch_000001.jsonl    # Individual batches
├── batch_000002.jsonl
├── ...
├── checkpoint.json       # Saved progress
└── complete_dataset.jsonl # Consolidated dataset (optional)
```

### Data Format

Each line in the `.jsonl` files has the format:

```json
{"text": "Complete example content here..."}
```

This format is **directly compatible** with the standard `tokenize_function` that looks for the `text` field.

## 🔄 Checkpoint System and Progress

The generator includes a robust checkpoint system and real-time monitoring:

### Automatic Checkpoints
- **Automatic saving**: Every 10,000 generated examples
- **Automatic recovery**: Resumes from the last checkpoint
- **Progress information**: Detailed tracking of advancement

### Real-time Monitoring
- **Detailed logs**: Information for each processed batch
- **Progress bar**: Continuous visual updates
- **Dynamic counters**: Generated examples and completion percentage
- **Visual indicators**: Emojis for easy identification (✓, 💾)

### Progress Output Example
```
2025-08-23 23:18:19,489 - INFO - Starting dataset generation: 10,000 examples
2025-08-23 23:18:21,279 - INFO - Ollama connection established
2025-08-23 23:18:22,156 - INFO - Processing batch 1/100
2025-08-23 23:18:25,789 - INFO - ✓ Saved batch 1: 100 elements | Total: 100
2025-08-23 23:18:26,234 - INFO - Batch 1 completed: 100 examples generated

Generating dataset: 15%|████████████                     | 15/100 batches [examples: 1,500, progress: 15.0%]

2025-08-23 23:25:34,123 - INFO - 💾 Checkpoint saved: 10,000 elements (100.0%)
```

### Checkpoint Format
```json
{
  "generated_count": 50000,
  "timestamp": 1692123456.789,
  "progress": 50.0
}
```

## 📈 Performance and Optimization

### Recommendations by Dataset Size

| Dataset Size | Batch Size | Concurrent | Estimated Time* |
|--------------|------------|------------|-----------------|
| 1K - 10K | 10-25 | 5-10 | 10-30 min |
| 10K - 100K | 25-100 | 10-20 | 1-5 hours |
| 100K - 1M | 100-200 | 20-30 | 5-20 hours |
| 1M+ | 200-500 | 30-50 | 20+ hours |

*Times depend on model, hardware, and Ollama configuration.

### Optimization Tips

1. **Adjust concurrency**: More concurrent tasks = higher memory usage
2. **Optimal batch size**: Balance between memory and network efficiency
3. **Appropriate model**: Smaller models = faster generation
4. **System resources**: Monitor CPU and memory during generation
5. **Custom timeouts**: Use `--timeout` for models requiring longer processing times

### Model-Specific Optimization

#### For CPU Execution
```bash
# Large models (30B+): Reduced concurrency and increased timeout
python main.py --model nemotron:latest --concurrent 5 --batch-size 50 --timeout 1800 --cpu-tips

# Medium models (7B-14B): Balanced settings
python main.py --model llama3.1 --concurrent 15 --batch-size 100 --timeout 300

# Small models: High concurrency
python main.py --model llama3.1 --concurrent 25 --batch-size 150 --timeout 120
```

#### For GPU Execution
```bash
# High-performance settings for GPU
python main.py --concurrent 50 --batch-size 500 --timeout 60
```

## 🧠 Token Context Optimization

The generator automatically detects and optimizes for different model context windows:

| Model Type | Context Window | Optimized Generation |
|------------|----------------|---------------------|
| **qwen3-coder** | 1024 tokens | 800 tokens |
| **nemotron** | 4096 tokens | 1500 tokens |
| **llama3.1** | 2048 tokens | 1500 tokens |
| **codellama** | 2048 tokens | 1500 tokens |
| **mistral** | 4096 tokens | 1500 tokens |

### Token Allocation Strategy
- **30% for prompts**: Reserved for instructions and context
- **70% for generation**: Available for content generation
- **Automatic detection**: Based on model name patterns

## 🐛 Troubleshooting

### Common Errors

#### "Cannot connect to host localhost:11434"
```bash
# Verify Ollama is running
ollama serve

# In another terminal, test connection
curl http://localhost:11434/api/tags
```

#### "Model not found"
```bash
# List available models
ollama list

# Download the necessary model
ollama pull llama3.1
ollama pull nemotron:latest
```

#### Model name typos (e.g., "nemotron:lastest")
The system now automatically suggests corrections:
```
ERROR - Model 'nemotron:lastest' not found.
ERROR - Did you mean? nemotron:latest
```

#### "Out of memory"
- Reduce `--concurrent` and `--batch-size`
- Use a smaller model
- Increase `--timeout` for CPU processing
- Close other memory-consuming applications

#### Very slow generation
- Check system resources (CPU, memory)
- Use a faster model (e.g., `llama3.1` vs `llama3.1:70b`)
- Adjust concurrency parameters
- Use `--cpu-tips` for model-specific recommendations

#### Timeout issues
```bash
# Use custom timeout for slow models or CPU processing
python main.py --timeout 1800 --model large_model

# Check automatic timeout recommendations
python main.py --cpu-tips --model your_model
```

#### Issues with specific languages
- **Wrong language content**: Verify the `--language` parameter
- **Inconsistent mixing**: In `mixed` mode, alternation is random by design
- **Specialized models**: Some models work better with specific languages:
  - `llama3.1`: Excellent for Spanish and English
  - `codellama`: Better for English code
  - `mistral`: Good for multilingual content
  - `nemotron`: Optimized for English technical content

## 🤝 Contributions

Contributions are welcome. Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-functionality`)
3. Commit your changes (`git commit -am 'Add new functionality'`)
4. Push to the branch (`git push origin feature/new-functionality`)
5. Create a Pull Request

## 📄 License

This project is under the MIT License. See `LICENSE` for more details.

## 🔗 Useful Links

- [Ollama Official Website](https://ollama.ai)
- [Ollama Models Library](https://ollama.ai/library)
- [Python AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)

---

## 💡 Tips and Best Practices

### For Multilingual Datasets
- **Mixed mode**: Ideal for training models that need to respond in both languages
- **Separate datasets**: For language-specific fine-tuning, generate individual datasets
- **Quality verification**: Review some examples to ensure language quality

### For Massive Datasets
- **Dedicated servers**: For very large datasets, use a server with good connectivity
- **Continuous monitoring**: Progress improvements allow you to monitor long generations
- **Checkpoints**: Automatic checkpoints allow resuming interrupted generations

### For Optimal Performance
- **Balanced concurrency**: More concurrent tasks = more memory, but also higher speed
- **Appropriate batch size**: Larger batches are more efficient but consume more memory
- **Right model**: Choose the model based on the type of content you need
- **Custom timeouts**: Use `--timeout` for models requiring special processing times

### CPU vs GPU Optimization
- **CPU execution**: Use lower concurrency and higher timeouts for large models
- **GPU execution**: Can handle higher concurrency and shorter timeouts
- **Mixed environments**: Adjust parameters based on your hardware setup

⚡ **Recommendation**: For production datasets, start with a small test using `--size 1000` to verify quality and performance before generating the complete dataset. Use `--cpu-tips` to get model-specific optimization recommendations.

## 🔧 Advanced Usage Examples

### High-Performance Production Setup
```bash
# Maximum performance on powerful hardware
python main.py --size 100000000 --model nemotron:latest --batch-size 1000 --concurrent 100 --language en --timeout 300 --output production_dataset
```

### CPU-Optimized Setup for Large Models
```bash
# Optimized for CPU processing of large models
python main.py --size 10000000 --model nemotron:latest --batch-size 50 --concurrent 5 --language en --timeout 1800 --cpu-tips
```

### Multi-Stage Generation
```bash
# Stage 1: Generate English content
python main.py --size 5000000 --language en --output stage1_english

# Stage 2: Generate Spanish content  
python main.py --size 5000000 --language es --output stage2_spanish

# Stage 3: Consolidate both datasets
python main.py --consolidate-only --output final_multilingual_dataset
```

### Quality-Focused Generation
```bash
# Lower concurrency for higher quality
python main.py --size 100000 --batch-size 25 --concurrent 5 --timeout 600 --language en --output high_quality_dataset
```