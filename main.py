#!/usr/bin/env python3
"""
Dataset Generator usando Ollama
Genera datasets masivos de hasta 100M de ejemplos de manera eficiente
"""

import json
import time
import random
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm.asyncio import tqdm

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuración para la generación del dataset"""
    target_size: int = 100_000_000  # 100M ejemplos
    batch_size: int = 100
    max_concurrent: int = 20
    output_dir: str = "generated_dataset"
    checkpoint_interval: int = 10000
    model_name: str = "llama3.1"
    ollama_url: str = "http://localhost:11434"
    language: str = "es"  # "es" para español, "en" para inglés, "mixed" para ambos
    max_tokens: Optional[int] = None  # Se calculará automáticamente
    timeout: Optional[int] = None  # Timeout en segundos, se calculará automáticamente si es None
    
    def get_optimized_concurrency(self) -> int:
        """Optimiza la concurrencia basada en el tamaño del modelo para CPU"""
        model_lower = self.model_name.lower()
        
        # Modelos muy grandes en CPU: reducir concurrencia para evitar saturar RAM
        if any(size in model_lower for size in ["30b", "32b", "34b", "70b"]):
            return min(self.max_concurrent, 5)  # Máximo 5 tareas concurrentes
        # Modelos medianos
        elif any(size in model_lower for size in ["7b", "8b", "13b", "14b"]):
            return min(self.max_concurrent, 10)  # Máximo 10 tareas concurrentes
        # Modelos pequeños
        else:
            return self.max_concurrent
    
    def get_model_context_length(self) -> int:
        """Detecta la longitud de contexto del modelo"""
        model_lower = self.model_name.lower()
        
        # Modelos conocidos con contextos específicos
        if "qwen" in model_lower and ("coder" in model_lower or "30b" in model_lower):
            return 1024  # qwen3-coder tiene 1024 tokens
        elif "nemotron" in model_lower:
            return 4096  # Nemotron típicamente 4096 tokens
        elif "codellama" in model_lower:
            return 2048  # CodeLlama típicamente 2048
        elif "llama" in model_lower:
            return 2048  # Llama3.1 típicamente 2048+
        elif "mistral" in model_lower:
            return 4096  # Mistral típicamente 4096
        else:
            return 2048  # Por defecto conservador
    
    def get_optimal_max_tokens(self) -> int:
        """Calcula tokens óptimos para generación basado en contexto del modelo"""
        context_length = self.get_model_context_length()
        
        # Reservar ~30% para el prompt, 70% para la respuesta
        prompt_reserve = int(context_length * 0.3)
        max_response = context_length - prompt_reserve
        
        # Nunca exceder límites seguros
        return min(max_response, 800 if context_length <= 1024 else 1500)

class OllamaClient:
    """Cliente para interactuar con Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1", timeout: Optional[int] = None):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.session = None
        
    def _get_timeout_for_model(self) -> int:
        """Calcula timeout dinámico basado en el tamaño del modelo o usa el timeout personalizado"""
        # Si hay timeout personalizado, usarlo
        if self.timeout is not None:
            return self.timeout
            
        # Sino, calcular automáticamente
        model_lower = self.model_name.lower()
        
        # Modelos muy grandes (30B+) en CPU
        if any(size in model_lower for size in ["30b", "32b", "34b", "70b"]):
            return 600  # 10 minutos para modelos grandes en CPU
        # Modelos medianos (7B-13B)
        elif any(size in model_lower for size in ["7b", "8b", "13b", "14b"]):
            return 180  # 3 minutos
        # Modelos pequeños
        else:
            return 120  # 2 minutos por defecto
        
    async def __aenter__(self):
        # Timeout dinámico basado en el tamaño del modelo
        timeout_seconds = self._get_timeout_for_model()
        logger.info(f"Configurando timeout de {timeout_seconds}s para modelo {self.model_name}")
        
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=50)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def verify_model(self) -> bool:
        """Verifica que el modelo existe en Ollama y sugiere alternativas"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["name"] for model in data.get("models", [])]
                    
                    if self.model_name in available_models:
                        return True
                    
                    # Buscar modelos similares para sugerir correcciones
                    similar_models = []
                    model_base = self.model_name.split(':')[0].lower()
                    
                    for model in available_models:
                        if model_base in model.lower():
                            similar_models.append(model)
                    
                    if similar_models:
                        logger.error(f"Modelo '{self.model_name}' no encontrado.")
                        logger.error(f"¿Quisiste decir? {', '.join(similar_models)}")
                    else:
                        logger.error(f"Modelo '{self.model_name}' no encontrado.")
                        logger.error(f"Modelos disponibles: {', '.join(available_models)}")
                    
                    return False
                else:
                    logger.error(f"Error al verificar modelos: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error al conectar con Ollama para verificar modelo: {e}")
            return False
    
    async def generate(self, prompt: str, max_tokens: int = 700, temperature: float = 0.8) -> Optional[str]:
        """Genera texto usando Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "response" in data:
                        return data["response"].strip()
                    else:
                        logger.error(f"Respuesta sin campo 'response': {data}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Error HTTP {response.status}: {error_text}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout al generar con modelo {self.model_name} - considera usar un modelo más rápido")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Error de conexión con Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado en generación: {type(e).__name__}: {e}")
            return None

class DatasetPrompts:
    """Plantillas de prompts para diferentes tipos de datasets"""
    
    @staticmethod
    def get_story_prompt(language: str = "es", max_tokens: int = 700) -> str:
        # Ajustar longitud según tokens disponibles
        if max_tokens <= 800:
            word_count = "200-300 palabras"
            word_count_en = "200-300 words"
        else:
            word_count = "400-500 palabras"
            word_count_en = "400-500 words"
            
        if language == "en":
            genres = ["sci-fi", "fantasy", "mystery", "romance", "horror", "adventure"]
            settings = ["future", "medieval", "city", "space", "forest"]
            
            genre = random.choice(genres)
            setting = random.choice(settings)
            
            return f"""Write a {genre} story in {setting}. Length: {word_count_en}. Include beginning, middle, end."""
        else:
            genres = ["ciencia ficción", "fantasía", "misterio", "romance", "terror", "aventura"]
            settings = ["futuro", "medieval", "ciudad", "espacio", "bosque"]
            
            genre = random.choice(genres)
            setting = random.choice(settings)
            
            return f"""Escribe un cuento de {genre} en {setting}. Longitud: {word_count}. Incluye inicio, desarrollo y final."""

    @staticmethod
    def get_instruction_prompt(language: str = "es", max_tokens: int = 700) -> str:
        word_limit = "150-250 palabras" if max_tokens <= 800 else "200-400 palabras"
        word_limit_en = "150-250 words" if max_tokens <= 800 else "200-400 words"
        
        if language == "en":
            tasks = [
                "cook pasta carbonara", "configure wifi router", "plant a garden",
                "explain Pythagorean theorem", "write a CV", "change a tire",
                "explain photosynthesis", "make origami"
            ]
            
            task = random.choice(tasks)
            
            return f"""How to {task}. Write clear steps. Length: {word_limit_en}."""
        else:
            tasks = [
                "cocinar pasta carbonara", "configurar router wifi", "plantar jardín",
                "explicar teorema Pitágoras", "escribir CV", "cambiar llanta",
                "explicar fotosíntesis", "hacer origami"
            ]
            
            task = random.choice(tasks)
            
            return f"""Cómo {task}. Escribe pasos claros. Longitud: {word_limit}."""

    @staticmethod
    def get_dialogue_prompt(language: str = "es", max_tokens: int = 700) -> str:
        exchanges = "6-8 intercambios" if max_tokens <= 800 else "8-10 intercambios"
        exchanges_en = "6-8 exchanges" if max_tokens <= 800 else "8-10 exchanges"
        
        if language == "en":
            scenarios = [
                "job interview", "friends planning travel", "medical consultation",
                "business meeting", "teacher-student", "family dinner"
            ]
            
            scenario = random.choice(scenarios)
            
            return f"""Write dialogue for {scenario}. {exchanges_en}. Be natural."""
        else:
            scenarios = [
                "entrevista trabajo", "amigos planificando viaje", "consulta médica",
                "reunión negocios", "profesor-estudiante", "cena familiar"
            ]
            
            scenario = random.choice(scenarios)
            
            return f"""Escribe diálogo para {scenario}. {exchanges}. Sé natural."""

    @staticmethod
    def get_article_prompt(language: str = "es", max_tokens: int = 700) -> str:
        length = "250-350 palabras" if max_tokens <= 800 else "400-500 palabras"
        length_en = "250-350 words" if max_tokens <= 800 else "400-500 words"
        
        if language == "en":
            topics = [
                "solar energy benefits", "biodiversity importance", "AI changing work",
                "chocolate history", "climate change effects", "color psychology",
                "video game evolution", "space mysteries"
            ]
            
            topic = random.choice(topics)
            
            return f"""Write article about {topic}. {length_en}. Educational tone."""
        else:
            topics = [
                "beneficios energía solar", "importancia biodiversidad", "IA cambiando trabajo",
                "historia chocolate", "efectos cambio climático", "psicología color",
                "evolución videojuegos", "misterios espacio"
            ]
            
            topic = random.choice(topics)
            
            return f"""Escribe artículo sobre {topic}. {length}. Tono educativo."""

    @staticmethod
    def get_code_prompt(language: str = "es", max_tokens: int = 700) -> str:
        prog_languages = ["Python", "JavaScript", "Java", "C++"]
        complexity = "simple" if max_tokens <= 800 else "completo"
        complexity_en = "simple" if max_tokens <= 800 else "complete"
        
        if language == "en":
            projects = [
                "calculator", "number guessing game", "password generator",
                "currency converter", "task list", "text analyzer"
            ]
            
            prog_language = random.choice(prog_languages)
            project = random.choice(projects)
            
            return f"""Write {complexity_en} {prog_language} code for {project}. Include comments."""
        else:
            projects = [
                "calculadora", "juego adivinanza", "generador contraseñas",
                "convertidor monedas", "lista tareas", "analizador texto"
            ]
            
            prog_language = random.choice(prog_languages)
            project = random.choice(projects)
            
            return f"""Escribe código {complexity} en {prog_language} para {project}. Incluye comentarios."""

    @staticmethod
    def get_essay_prompt(language: str = "es", max_tokens: int = 700) -> str:
        length = "250-350 palabras" if max_tokens <= 800 else "400-500 palabras"
        length_en = "250-350 words" if max_tokens <= 800 else "400-500 words"
        
        if language == "en":
            themes = [
                "education importance", "social media impact", "AI ethics",
                "remote work future", "environmental conservation", "music influence",
                "globalization challenges", "digital reading"
            ]
            
            theme = random.choice(themes)
            
            return f"""Write essay about {theme}. {length_en}. Academic tone."""
        else:
            themes = [
                "importancia educación", "impacto redes sociales", "ética IA",
                "futuro trabajo remoto", "conservación ambiental", "influencia música",
                "desafíos globalización", "lectura digital"
            ]
            
            theme = random.choice(themes)
            
            return f"""Escribe ensayo sobre {theme}. {length}. Tono académico."""

class DatasetGenerator:
    """Generador principal del dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.prompts = DatasetPrompts()
        self.generated_count = 0
        self.current_batch = []
        
    def get_random_prompt(self) -> str:
        """Obtiene un prompt aleatorio optimizado para tokens"""
        # Seleccionar idioma para este prompt
        if self.config.language == "mixed":
            current_lang = random.choice(["es", "en"])
        else:
            current_lang = self.config.language
        
        # Obtener tokens óptimos para el modelo
        max_tokens = self.config.max_tokens or self.config.get_optimal_max_tokens()
            
        prompt_methods = [
            self.prompts.get_story_prompt,
            self.prompts.get_instruction_prompt,
            self.prompts.get_dialogue_prompt,
            self.prompts.get_article_prompt,
            self.prompts.get_code_prompt,
            self.prompts.get_essay_prompt
        ]
        return random.choice(prompt_methods)(current_lang, max_tokens)
    
    async def generate_batch(self, client: OllamaClient, batch_id: int) -> List[Dict[str, Any]]:
        """Genera un lote de ejemplos"""
        batch_results = []
        
        # Obtener tokens optimizados para este modelo
        optimal_tokens = self.config.max_tokens or self.config.get_optimal_max_tokens()
        
        tasks = []
        for _ in range(self.config.batch_size):
            prompt = self.get_random_prompt()
            tasks.append(client.generate(prompt, max_tokens=optimal_tokens))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, str) and result:
                # Siempre crear formato simple con campo "text"
                batch_results.append({
                    "text": result.strip()
                })
        
        return batch_results
    
    def save_batch(self, batch_data: List[Dict[str, Any]], batch_id: int):
        """Guarda un lote al disco"""
        filename = self.output_dir / f"batch_{batch_id:06d}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.generated_count += len(batch_data)
        logger.info(f"✓ Guardado lote {batch_id + 1}: {len(batch_data)} elementos | Total: {self.generated_count:,}")
    
    def save_checkpoint(self):
        """Guarda un checkpoint del progreso"""
        checkpoint_data = {
            "generated_count": self.generated_count,
            "timestamp": time.time(),
            "progress": (self.generated_count / self.config.target_size) * 100
        }
        
        with open(self.output_dir / "checkpoint.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"💾 Checkpoint guardado: {self.generated_count:,} elementos ({(self.generated_count / self.config.target_size) * 100:.1f}%)")
    
    def load_checkpoint(self) -> int:
        """Carga el último checkpoint"""
        checkpoint_file = self.output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                self.generated_count = data.get("generated_count", 0)
                logger.info(f"Checkpoint cargado: {self.generated_count:,} elementos")
                return self.generated_count // self.config.batch_size
        return 0
    
    async def generate_dataset(self):
        """Función principal de generación"""
        logger.info(f"Iniciando generación de dataset: {self.config.target_size:,} ejemplos")
        
        start_batch = self.load_checkpoint()
        total_batches = (self.config.target_size + self.config.batch_size - 1) // self.config.batch_size
        
        async with OllamaClient(self.config.ollama_url, self.config.model_name, self.config.timeout) as client:
            # Verificar que el modelo existe
            if not await client.verify_model():
                logger.error(f"Modelo '{self.config.model_name}' no encontrado. Usa 'ollama pull {self.config.model_name}' para descargarlo")
                return
            
            # Test de conexión
            logger.info(f"Probando conexión con modelo {self.config.model_name}...")
            test_result = await client.generate("Test", max_tokens=5)
            if not test_result:
                logger.error("No se pudo generar contenido con el modelo")
                return
            
            logger.info(f"Conexión con Ollama establecida usando {self.config.model_name}")
            
            # Configuración optimizada para tokens
            context_length = self.config.get_model_context_length()
            optimal_tokens = self.config.max_tokens or self.config.get_optimal_max_tokens()
            logger.info(f"Contexto del modelo: {context_length} tokens, generación optimizada: {optimal_tokens} tokens")
            
            # Semáforo para controlar concurrencia optimizada para CPU
            optimized_concurrency = self.config.get_optimized_concurrency()
            logger.info(f"Usando concurrencia optimizada: {optimized_concurrency} tareas simultáneas")
            semaphore = asyncio.Semaphore(optimized_concurrency)
            
            async def process_batch(batch_id: int):
                async with semaphore:
                    logger.info(f"Procesando lote {batch_id + 1}/{total_batches}")
                    batch_data = await self.generate_batch(client, batch_id)
                    if batch_data:
                        self.save_batch(batch_data, batch_id)
                        logger.info(f"Lote {batch_id + 1} completado: {len(batch_data)} ejemplos generados")
                        
                        if batch_id % (self.config.checkpoint_interval // self.config.batch_size) == 0:
                            self.save_checkpoint()
                    
                    return len(batch_data)
            
            # Genera todos los lotes
            tasks = []
            for batch_id in range(start_batch, total_batches):
                if self.generated_count >= self.config.target_size:
                    break
                
                tasks.append(process_batch(batch_id))
            
            # Ejecuta con barra de progreso
            results = []
            completed = 0
            with tqdm(total=len(tasks), desc="Generando dataset", unit="lotes") as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'ejemplos': f"{sum(results):,}",
                        'progreso': f"{(sum(results) / self.config.target_size) * 100:.1f}%"
                    })
        
        self.save_checkpoint()
        logger.info(f"Generación completada: {self.generated_count:,} elementos generados")
    
    def consolidate_dataset(self):
        """Consolida todos los archivos en uno solo"""
        logger.info("Consolidando dataset...")
        
        output_file = self.output_dir / "complete_dataset.jsonl"
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for batch_file in batch_files:
                with open(batch_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
        
        # Estadísticas finales
        total_lines = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
        file_size = output_file.stat().st_size / (1024**3)  # GB
        
        logger.info(f"Dataset consolidado:")
        logger.info(f"  - Archivo: {output_file}")
        logger.info(f"  - Elementos: {total_lines:,}")
        logger.info(f"  - Tamaño: {file_size:.2f} GB")

def show_cpu_optimization_tips(model_name: str):
    """Muestra consejos de optimización para modelos en CPU"""
    model_lower = model_name.lower()
    
    # Crear config temporal para obtener información del modelo
    temp_config = DatasetConfig(model_name=model_name)
    context_length = temp_config.get_model_context_length()
    optimal_tokens = temp_config.get_optimal_max_tokens()
    
    print(f"\n🖥️  Recomendaciones para {model_name} en CPU:")
    print("=" * 50)
    print(f"📏 Contexto del modelo: {context_length} tokens")
    print(f"⚡ Generación optimizada: {optimal_tokens} tokens")
    print()
    
    if any(size in model_lower for size in ["30b", "32b", "34b", "70b"]):
        print("📊 Modelo muy grande detectado (30B+)")
        print("💡 Configuración recomendada:")
        print("   --concurrent 3-5    (Reducir concurrencia)")
        print("   --batch-size 25-50  (Lotes más pequeños)")
        print(f"   Ejemplo: python main.py --model {model_name} --concurrent 3 --batch-size 25 --size 1000")
        print("\n⚡ Tips adicionales:")
        print("   • Cierra otras aplicaciones para liberar RAM")
        print("   • Usa OLLAMA_NUM_PARALLEL=1 para limitar instancias")
        print("   • Considera usar un modelo más pequeño para mayor velocidad")
        
    elif any(size in model_lower for size in ["7b", "8b", "13b", "14b"]):
        print("📊 Modelo mediano detectado (7B-14B)")
        print("💡 Configuración recomendada:")
        print("   --concurrent 5-10   (Concurrencia moderada)")
        print("   --batch-size 50-100 (Lotes estándar)")
        print(f"   Ejemplo: python main.py --model {model_name} --concurrent 8 --batch-size 75 --size 10000")
        
    else:
        print("📊 Modelo pequeño/estándar detectado")
        print("💡 Configuración recomendada:")
        print("   --concurrent 10-20  (Concurrencia alta)")
        print("   --batch-size 100+   (Lotes grandes)")
        print(f"   Ejemplo: python main.py --model {model_name} --concurrent 15 --batch-size 100 --size 50000")
    
    # Crear cliente temporal para obtener timeout
    temp_client = OllamaClient(model_name=model_name)
    auto_timeout = temp_client._get_timeout_for_model()
    print(f"\n⏱️  Timeout automático: {auto_timeout} segundos")
    print(f"    Usar --timeout X para personalizar")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generador de dataset masivo")
    parser.add_argument("--size", type=int, default=100_000_000, help="Tamaño objetivo del dataset")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamaño del lote")
    parser.add_argument("--concurrent", type=int, default=20, help="Máximo de tareas concurrentes")
    parser.add_argument("--output", type=str, default="generated_dataset", help="Directorio de salida")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL de Ollama")
    parser.add_argument("--model", type=str, default="llama3.1", help="Modelo de Ollama a usar")
    parser.add_argument("--language", type=str, default="es", choices=["es", "en", "mixed"], help="Idioma del dataset: es (español), en (inglés), mixed (ambos)")
    parser.add_argument("--consolidate-only", action="store_true", help="Solo consolidar archivos existentes")
    parser.add_argument("--cpu-tips", action="store_true", help="Muestra consejos de optimización para CPU")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout en segundos para cada generación (automático si no se especifica)")
    
    args = parser.parse_args()
    
    # Mostrar tips de optimización para CPU si se solicita
    if args.cpu_tips:
        show_cpu_optimization_tips(args.model)
        return
    
    config = DatasetConfig(
        target_size=args.size,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent,
        output_dir=args.output,
        model_name=args.model,
        ollama_url=args.ollama_url,
        language=args.language,
        timeout=args.timeout
    )
    
    generator = DatasetGenerator(config)
    
    if args.consolidate_only:
        generator.consolidate_dataset()
    else:
        # Ejecuta la generación
        asyncio.run(generator.generate_dataset())
        
        # Consolida al final
        consolidate = input("¿Consolidar dataset en un solo archivo? (y/N): ")
        if consolidate.lower() == 'y':
            generator.consolidate_dataset()

if __name__ == "__main__":
    main()