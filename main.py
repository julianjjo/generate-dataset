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

class OllamaClient:
    """Cliente para interactuar con Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=50)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
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
                    return data["response"].strip()
                else:
                    logger.error(f"Error en API: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error en generación: {e}")
            return None

class DatasetPrompts:
    """Plantillas de prompts para diferentes tipos de datasets"""
    
    @staticmethod
    def get_story_prompt() -> str:
        genres = ["ciencia ficción", "fantasía", "misterio", "romance", "terror", "aventura", "drama"]
        settings = ["futuro", "pasado medieval", "ciudad moderna", "espacio", "bosque encantado", "laboratorio"]
        
        genre = random.choice(genres)
        setting = random.choice(settings)
        
        return f"""Escribe un cuento completo de {genre} ambientado en {setting}. 
El cuento debe tener entre 300-500 palabras, con inicio, desarrollo y final.
Solo responde con el texto del cuento, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_instruction_prompt() -> str:
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
    def get_dialogue_prompt() -> str:
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
    def get_article_prompt() -> str:
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
    def get_code_prompt() -> str:
        languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
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
        
        language = random.choice(languages)
        project = random.choice(projects)
        
        return f"""Crea código completo en {language} para {project}.
Incluye comentarios explicativos, manejo básico de errores y ejemplos de uso.
El código debe ser funcional y bien estructurado.
Solo responde con el código y comentarios, sin formato JSON ni etiquetas adicionales."""

    @staticmethod
    def get_essay_prompt() -> str:
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
        prompt_methods = [
            self.prompts.get_story_prompt,
            self.prompts.get_instruction_prompt,
            self.prompts.get_dialogue_prompt,
            self.prompts.get_article_prompt,
            self.prompts.get_code_prompt,
            self.prompts.get_essay_prompt
        ]
        return random.choice(prompt_methods)()
    
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
        logger.info(f"Guardado lote {batch_id}: {len(batch_data)} elementos")
    
    def save_checkpoint(self):
        """Guarda un checkpoint del progreso"""
        checkpoint_data = {
            "generated_count": self.generated_count,
            "timestamp": time.time(),
            "progress": (self.generated_count / self.config.target_size) * 100
        }
        
        with open(self.output_dir / "checkpoint.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint guardado: {self.generated_count:,} elementos generados")
    
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
            # Test de conexión
            test_result = await client.generate("Test de conexión", max_tokens=10)
            if not test_result:
                logger.error("No se pudo conectar con Ollama")
                return
            
            logger.info("Conexión con Ollama establecida")
            
            # Semáforo para controlar concurrencia
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            async def process_batch(batch_id: int):
                async with semaphore:
                    batch_data = await self.generate_batch(client, batch_id)
                    if batch_data:
                        self.save_batch(batch_data, batch_id)
                        
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
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generando dataset"):
                result = await task
                results.append(result)
        
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

def main():
    parser = argparse.ArgumentParser(description="Generador de dataset masivo")
    parser.add_argument("--size", type=int, default=100_000_000, help="Tamaño objetivo del dataset")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamaño del lote")
    parser.add_argument("--concurrent", type=int, default=20, help="Máximo de tareas concurrentes")
    parser.add_argument("--output", type=str, default="generated_dataset", help="Directorio de salida")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL de Ollama")
    parser.add_argument("--model", type=str, default="llama3.1", help="Modelo de Ollama a usar")
    parser.add_argument("--consolidate-only", action="store_true", help="Solo consolidar archivos existentes")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        target_size=args.size,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent,
        output_dir=args.output,
        model_name=args.model,
        ollama_url=args.ollama_url
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