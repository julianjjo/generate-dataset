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

class OllamaClient:
    """Cliente para interactuar con Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = None
        
    def _get_timeout_for_model(self) -> int:
        """Calcula timeout dinámico basado en el tamaño del modelo"""
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
        """Verifica que el modelo existe en Ollama"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.model_name in models
                else:
                    logger.error(f"Error al verificar modelos: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error al conectar con Ollama para verificar modelo: {e}")
            return False
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.8) -> Optional[str]:
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
    def get_story_prompt(language: str = "es") -> str:
        if language == "en":
            genres = ["science fiction", "fantasy", "mystery", "romance", "horror", "adventure", "drama"]
            settings = ["future", "medieval past", "modern city", "space", "enchanted forest", "laboratory"]
            
            genre = random.choice(genres)
            setting = random.choice(settings)
            
            return f"""Write a complete {genre} story set in {setting}. 
The story should be 300-500 words long, with beginning, development and ending.
Only respond with the story text, no JSON format or additional tags."""
        else:
            genres = ["ciencia ficción", "fantasía", "misterio", "romance", "terror", "aventura", "drama"]
            settings = ["futuro", "pasado medieval", "ciudad moderna", "espacio", "bosque encantado", "laboratorio"]
            
            genre = random.choice(genres)
            setting = random.choice(settings)
            
            return f"""Escribe un cuento completo de {genre} ambientado en {setting}. 
El cuento debe tener entre 300-500 palabras, con inicio, desarrollo y final.
Solo responde con el texto del cuento, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_instruction_prompt(language: str = "es") -> str:
        if language == "en":
            tasks = [
                "explain how to cook pasta carbonara",
                "teach how to configure a wifi router",
                "show how to plant a garden",
                "explain the Pythagorean theorem",
                "teach how to write a professional CV",
                "show how to change a tire",
                "explain how photosynthesis works",
                "teach basic origami"
            ]
            
            task = random.choice(tasks)
            
            return f"""Create a complete instruction to {task}.
Include an introduction, detailed numbered steps, useful tips and a conclusion.
The text should be clear, educational and at least 200 words long.
Only respond with the instructional text, no JSON format or additional tags."""
        else:
            tasks = [
                "explicar cómo cocinar pasta carbonara",
                "enseñar a configurar un router wifi",
                "mostrar cómo plantar un jardín",
                "explicar el teorema de Pitágoras",
                "enseñar a escribir un CV profesional",
                "mostrar cómo cambiar una llanta",
                "explicar cómo funciona la fotosíntesis",
                "enseñar a hacer origami básico"
            ]
            
            task = random.choice(tasks)
            
            return f"""Crea una instrucción completa para {task}.
Incluye una introducción, pasos detallados numerados, consejos útiles y una conclusión.
El texto debe ser claro, educativo y de al menos 200 palabras.
Solo responde con el texto instructivo, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_dialogue_prompt(language: str = "es") -> str:
        if language == "en":
            scenarios = [
                "a job interview",
                "a discussion between friends about travel plans",
                "a medical consultation",
                "a business negotiation",
                "a class between teacher and student",
                "a family conversation at dinner"
            ]
            
            scenario = random.choice(scenarios)
            
            return f"""Write a natural dialogue for {scenario}.
The dialogue should have at least 8-10 exchanges, be realistic and show different personalities.
Include brief action descriptions between dialogue lines.
Only respond with the complete dialogue, no JSON format or additional tags."""
        else:
            scenarios = [
                "una entrevista de trabajo",
                "una discusión entre amigos sobre planes de viaje",
                "una consulta médica",
                "una negociación comercial",
                "una clase entre profesor y estudiante",
                "una conversación familiar en la cena"
            ]
            
            scenario = random.choice(scenarios)
            
            return f"""Escribe un diálogo natural para {scenario}.
El diálogo debe tener al menos 8-10 intercambios, ser realista y mostrar personalidades diferentes.
Incluye descripciones breves de acciones entre las líneas de diálogo.
Solo responde with el diálogo completo, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_article_prompt(language: str = "es") -> str:
        if language == "en":
            topics = [
                "the benefits of solar energy",
                "the importance of biodiversity",
                "how artificial intelligence is changing work",
                "the history of chocolate",
                "the effects of climate change on oceans",
                "the psychology of color in marketing",
                "the evolution of video games",
                "the mysteries of deep space"
            ]
            
            topic = random.choice(topics)
            
            return f"""Write an informative article about {topic}.
The article should be 400-600 words long, with title, introduction, development with subtopics and conclusion.
Use an educational but accessible tone for the general public.
Only respond with the complete article, no JSON format or additional tags."""
        else:
            topics = [
                "los beneficios de la energía solar",
                "la importancia de la biodiversidad",
                "cómo la inteligencia artificial está cambiando el trabajo",
                "la historia del chocolate",
                "los efectos del cambio climático en los océanos",
                "la psicología del color en el marketing",
                "la evolución de los videojuegos",
                "los misterios del espacio profundo"
            ]
            
            topic = random.choice(topics)
            
            return f"""Escribe un artículo informativo sobre {topic}.
El artículo debe tener entre 400-600 palabras, con título, introducción, desarrollo con subtemas y conclusión.
Usa un tono educativo pero accesible para el público general.
Solo responde con el artículo completo, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_code_prompt(language: str = "es") -> str:
        prog_languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
        
        if language == "en":
            projects = [
                "a library management system",
                "a number guessing game",
                "a basic calculator",
                "a simple login system",
                "a password generator",
                "a currency converter",
                "a task organizer",
                "a text analyzer"
            ]
            
            prog_language = random.choice(prog_languages)
            project = random.choice(projects)
            
            return f"""Create complete code in {prog_language} for {project}.
Include explanatory comments, basic error handling and usage examples.
The code should be functional and well-structured.
Only respond with the code and comments, no JSON format or additional tags."""
        else:
            projects = [
                "un sistema de gestión de biblioteca",
                "un juego de adivinanza de números",
                "un calculadora básica",
                "un sistema de login simple",
                "un generador de contraseñas",
                "un convertidor de monedas",
                "un organizador de tareas",
                "un analizador de texto"
            ]
            
            prog_language = random.choice(prog_languages)
            project = random.choice(projects)
            
            return f"""Crea código completo en {prog_language} para {project}.
Incluye comentarios explicativos, manejo básico de errores y ejemplos de uso.
El código debe ser funcional y bien estructurado.
Solo responde con el código y comentarios, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_essay_prompt(language: str = "es") -> str:
        if language == "en":
            themes = [
                "the importance of education in the 21st century",
                "the impact of social media on human relationships",
                "ethics in artificial intelligence",
                "the future of remote work",
                "environmental conservation",
                "the influence of music on mood",
                "the challenges of globalization",
                "the importance of reading in the digital age"
            ]
            
            theme = random.choice(themes)
            
            return f"""Write a reflective essay about {theme}.
The essay should be 400-500 words long, with a clear thesis, solid arguments and examples.
Use an academic but accessible tone.
Only respond with the complete essay, no JSON format or additional tags."""
        else:
            themes = [
                "la importancia de la educación en el siglo XXI",
                "el impacto de las redes sociales en las relaciones humanas",
                "la ética en la inteligencia artificial",
                "el futuro del trabajo remoto",
                "la conservación del medio ambiente",
                "la influencia de la música en el estado de ánimo",
                "los desafíos de la globalización",
                "la importancia de la lectura en la era digital"
            ]
            
            theme = random.choice(themes)
            
            return f"""Escribe un ensayo reflexivo sobre {theme}.
El ensayo debe tener entre 400-500 palabras, con una tesis clara, argumentos sólidos y ejemplos.
Usa un tono académico pero accesible.
Solo responde con el ensayo completo, sin formato JSON ni etiquetas adicionales."""

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
        """Obtiene un prompt aleatorio"""
        # Seleccionar idioma para este prompt
        if self.config.language == "mixed":
            current_lang = random.choice(["es", "en"])
        else:
            current_lang = self.config.language
            
        prompt_methods = [
            self.prompts.get_story_prompt,
            self.prompts.get_instruction_prompt,
            self.prompts.get_dialogue_prompt,
            self.prompts.get_article_prompt,
            self.prompts.get_code_prompt,
            self.prompts.get_essay_prompt
        ]
        return random.choice(prompt_methods)(current_lang)
    
    async def generate_batch(self, client: OllamaClient, batch_id: int) -> List[Dict[str, Any]]:
        """Genera un lote de ejemplos"""
        batch_results = []
        
        tasks = []
        for _ in range(self.config.batch_size):
            prompt = self.get_random_prompt()
            tasks.append(client.generate(prompt))
        
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
        
        async with OllamaClient(self.config.ollama_url, self.config.model_name) as client:
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
    
    print(f"\n🖥️  Recomendaciones para {model_name} en CPU:")
    print("=" * 50)
    
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
    
    print(f"\n⏱️  Timeout automático: {600 if any(size in model_lower for size in ['30b', '32b', '34b', '70b']) else 180 if any(size in model_lower for size in ['7b', '8b', '13b', '14b']) else 120} segundos")
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
        language=args.language
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