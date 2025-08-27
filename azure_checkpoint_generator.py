#!/usr/bin/env python3
"""
Dataset Generator AGRESIVO para Azure AI con grok-3-mini usando API REST
- Optimizado para 50,000 requests/minuto usando grok-3-mini
- Concurrencia máxima: 50 requests simultáneos
- Batch size agresivo: 100 ejemplos por lote
- Delays mínimos: 50ms entre requests, 100ms entre batches
- Checkpoint automático cada 1000 ejemplos
- Recuperación automática en caso de interrupciones
- Progress tracking detallado
- Sistema de recovery ultra robusto
- API REST nativa para compatibilidad total con grok-3-mini
- Formato CHAT optimizado (no instruct) - combina system+user en un mensaje

Cambios de optimización:
- Modelo: Llama-3.3-70B-Instruct → grok-3-mini (formato chat)
- Cliente: Azure SDK → API REST con aiohttp
- Prompts: System+User separados → Combinados en mensaje user único
- request_delay: 2.0s → 0.05s (40x más rápido)
- batch_delay: 5.0s → 0.1s (50x más rápido)
- max_concurrent: 2 → 50 (25x más concurrencia)
- batch_size: 10 → 100 (10x más elementos por batch)
- retry_delay: 3.0s → 0.5s (6x más rápido en retries)
- timeout: 600s → 300s (fallar rápido)
- max_completion_tokens en lugar de max_tokens para grok-3-mini
"""

import json
import time
import random
import asyncio
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import aiohttp
# from azure.core.exceptions import AzureError  # No usado actualmente

# Configuración de logging
log_level = logging.DEBUG if os.getenv("DEBUG_AZURE_CLIENT") else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuración AGRESIVA optimizada para 50k requests/min"""
    target_size: int = 1000
    batch_size: int = 100  # 10x más agresivo
    max_concurrent: int = 50  # 25x más concurrencia
    output_dir: str = "checkpoint_dataset"
    model_name: str = "grok-3-mini"
    azure_endpoint: str = "https://hrmllma.services.ai.azure.com/models"
    api_version: str = "2024-05-01-preview"
    language: str = "es"
    max_tokens: int = 1500
    temperature: float = 0.8
    top_p: float = 0.1
    
    # Sistema de checkpoints
    checkpoint_interval: int = 1000  # Checkpoint cada 1000 ejemplos
    mini_checkpoint_interval: int = 100  # Mini-checkpoint cada 100
    progress_save_interval: int = 50  # Guardar progreso cada 50
    
    # Rate limiting AGRESIVO - optimizado para 50k requests/min
    request_delay: float = 0.05  # 40x más rápido (72ms entre requests)
    batch_delay: float = 0.1     # 50x más rápido
    retry_attempts: int = 16      # Más intentos para manejar el alto volumen
    retry_delay: float = 0.5     # 6x más rápido en retries
    timeout: int = 300           # Timeout más bajo para fallar rápido

class CheckpointManager:
    """Gestor robusto de checkpoints y recuperación"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Archivos del sistema de checkpoints
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.progress_file = self.output_dir / "progress.json"
        self.recovery_file = self.output_dir / "recovery.json"
        self.session_file = self.output_dir / "session.json"
        
        # Estado interno
        self.last_checkpoint_count = 0
        self.session_start_time = time.time()
        self.session_id = int(self.session_start_time)
    
    def save_checkpoint(self, generated_count: int, batch_id: int, config: CheckpointConfig, extra_info: dict = None):
        """Guardar checkpoint completo con toda la información necesaria"""
        checkpoint_data = {
            "version": "1.0",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "generated_count": generated_count,
            "batch_id": batch_id,
            "target_size": config.target_size,
            "progress_pct": (generated_count / config.target_size) * 100,
            
            # Configuración completa para recuperación exacta
            "config": {
                "batch_size": config.batch_size,
                "max_concurrent": config.max_concurrent,
                "model_name": config.model_name,
                "language": config.language,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "request_delay": config.request_delay,
                "batch_delay": config.batch_delay,
                "checkpoint_interval": config.checkpoint_interval
            },
            
            # Estadísticas de sesión
            "session_stats": {
                "start_time": self.session_start_time,
                "elapsed_hours": (time.time() - self.session_start_time) / 3600,
                "avg_examples_per_hour": generated_count / ((time.time() - self.session_start_time) / 3600) if generated_count > 0 else 0
            },
            
            # Información adicional
            "checkpoint_reason": extra_info.get("reason", "automatic") if extra_info else "automatic",
            "last_batch_size": extra_info.get("batch_size", config.batch_size) if extra_info else config.batch_size,
            "estimated_remaining_hours": self.estimate_remaining_time(generated_count, config.target_size)
        }
        
        # Guardar checkpoint principal
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Guardar backup con timestamp
        backup_file = self.output_dir / f"checkpoint_{int(time.time())}.json"
        with open(backup_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.last_checkpoint_count = generated_count
        
        logger.info(f"💾 CHECKPOINT guardado: {generated_count:,} ejemplos ({checkpoint_data['progress_pct']:.1f}%)")
        
        # Limpiar backups antiguos (mantener solo los últimos 10)
        self.cleanup_old_backups()
        
        return checkpoint_data
    
    def save_progress(self, generated_count: int, batch_id: int, recent_stats: dict = None):
        """Guardar progreso ligero (más frecuente)"""
        progress_data = {
            "timestamp": time.time(),
            "generated_count": generated_count,
            "batch_id": batch_id,
            "session_id": self.session_id,
            "recent_stats": recent_stats or {}
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def save_recovery_info(self, generated_count: int, batch_id: int, error_info: str = None):
        """Guardar información detallada para recuperación"""
        recovery_data = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "last_successful_count": generated_count,
            "last_successful_batch": batch_id,
            "error_info": error_info,
            "recovery_instructions": {
                "resume_from_batch": batch_id + 1,
                "expected_count": generated_count,
                "files_to_check": [f"batch_{i:06d}.jsonl" for i in range(batch_id + 1)]
            }
        }
        
        with open(self.recovery_file, 'w') as f:
            json.dump(recovery_data, f, indent=2)
        
        logger.info(f"🛠️ Recovery info guardada: {generated_count} ejemplos, batch {batch_id}")
    
    def load_checkpoint(self) -> tuple[int, int, dict]:
        """Cargar checkpoint con validación robusta"""
        generated_count = 0
        batch_id = 0
        checkpoint_data = {}
        
        # Intentar cargar checkpoint principal
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                generated_count = checkpoint_data.get("generated_count", 0)
                batch_id = checkpoint_data.get("batch_id", 0)
                
                logger.info(f"📥 CHECKPOINT cargado:")
                logger.info(f"   📊 Ejemplos: {generated_count:,}")
                logger.info(f"   🎯 Lote: {batch_id}")
                logger.info(f"   📅 Fecha: {time.ctime(checkpoint_data.get('timestamp', 0))}") 
                logger.info(f"   ⏱️ Tiempo restante estimado: {checkpoint_data.get('estimated_remaining_hours', 0):.1f}h")
                
                # Validar consistencia
                if self.validate_checkpoint_consistency(generated_count, batch_id, checkpoint_data.get("config", {})):
                    return generated_count, batch_id, checkpoint_data
                else:
                    logger.warning("⚠️ Checkpoint inconsistente, intentando recuperar...")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando checkpoint: {e}")
        
        # Fallback: intentar recovery
        if self.recovery_file.exists():
            try:
                with open(self.recovery_file, 'r') as f:
                    recovery_data = json.load(f)
                
                recovery_count = recovery_data.get("last_successful_count", 0)
                recovery_batch = recovery_data.get("last_successful_batch", 0)
                
                logger.info(f"🛠️ RECOVERY INFO cargada:")
                logger.info(f"   📊 Último conteo exitoso: {recovery_count:,}")
                logger.info(f"   ❗ Último error: {recovery_data.get('error_info', 'N/A')}")
                
                if self.validate_files_exist(recovery_batch):
                    return recovery_count, recovery_batch, recovery_data
                    
            except Exception as e:
                logger.error(f"❌ Error cargando recovery: {e}")
        
        # Fallback: contar archivos existentes
        existing_count, existing_batch = self.count_existing_files()
        if existing_count > 0:
            logger.info(f"🔍 Detectados {existing_count:,} ejemplos en {existing_batch} archivos de lote")
            return existing_count, existing_batch, {}
        
        logger.info("🚀 Comenzando desde cero")
        return 0, 0, {}
    
    def validate_checkpoint_consistency(self, count: int, batch_id: int, config: dict) -> bool:
        """Validar que el checkpoint sea consistente"""
        expected_batches = count // config.get("batch_size", 10)
        return abs(batch_id - expected_batches) <= 2  # Tolerancia de 2 lotes
    
    def validate_files_exist(self, up_to_batch: int) -> bool:
        """Verificar que los archivos de lotes existan"""
        missing_files = []
        for i in range(up_to_batch):
            batch_file = self.output_dir / f"batch_{i:06d}.jsonl"
            if not batch_file.exists():
                missing_files.append(i)
        
        if missing_files:
            logger.warning(f"⚠️ Archivos de lote faltantes: {missing_files}")
            return len(missing_files) < 3  # Permitir hasta 3 archivos faltantes
        
        return True
    
    def count_existing_files(self) -> tuple[int, int]:
        """Contar ejemplos en archivos existentes"""
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        total_count = 0
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    total_count += sum(1 for _ in f)
            except Exception as e:
                logger.warning(f"⚠️ Error leyendo {batch_file}: {e}")
        
        return total_count, len(batch_files)
    
    def cleanup_old_backups(self):
        """Limpiar backups antiguos"""
        backup_files = sorted(self.output_dir.glob("checkpoint_*.json"))
        if len(backup_files) > 10:
            for old_file in backup_files[:-10]:
                old_file.unlink()
    
    def estimate_remaining_time(self, generated_count: int, target_size: int) -> float:
        """Estimar tiempo restante conservadoramente"""
        if generated_count <= 0:
            return float('inf')
        
        elapsed_hours = (time.time() - self.session_start_time) / 3600
        rate_per_hour = generated_count / elapsed_hours if elapsed_hours > 0 else 50
        remaining = target_size - generated_count
        
        return remaining / rate_per_hour if rate_per_hour > 0 else float('inf')

class ConservativeAzureClient:
    """Cliente Azure con rate limiting AGRESIVO usando API REST para grok-3-mini"""
    
    def __init__(self, endpoint: str, api_key: str, model_name: str, config: CheckpointConfig):
        self.endpoint = endpoint
        self.model_name = model_name
        self.config = config
        self.api_key = api_key
        self.session = None
        self.stats = {
            'requests_total': 0,
            'successful': 0,
            'failed': 0,
            'rate_limited': 0
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
        logger.info(f"📊 Stats de sesión: {self.stats['successful']}/{self.stats['requests_total']} exitosos")
    
    async def generate_safely(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generación optimizada para alto volumen usando API REST - formato chat"""
        self.stats['requests_total'] += 1
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"🔄 Iniciando intento {attempt + 1}/{self.config.retry_attempts}")
                
                # Delay mínimo entre requests para maximizar throughput
                await asyncio.sleep(self.config.request_delay)
                
                # Combinar system y user prompt en un solo mensaje para grok-3-mini (formato chat)
                combined_content = f"{system_prompt}\n\n{user_prompt}"
                
                # Preparar payload para grok-3-mini (solo user role)
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": combined_content
                        }
                    ],
                    "max_completion_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "model": self.model_name
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                api_url = f"{self.endpoint}/chat/completions?api-version={self.config.api_version}"
                
                logger.debug(f"📡 Enviando request a API (intento {attempt + 1})")
                
                # Agregar timeout específico para cada request
                timeout = aiohttp.ClientTimeout(total=30)  # 30s timeout por request
                async with self.session.post(api_url, json=payload, headers=headers, timeout=timeout) as response:
                    logger.debug(f"📥 Respuesta recibida: status {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            content = data["choices"][0]["message"]["content"].strip()
                            if len(content) > 20:
                                self.stats['successful'] += 1
                                logger.debug(f"✅ Request exitoso (intento {attempt + 1})")
                                return content
                        
                        logger.warning(f"⚠️ Respuesta vacía o inválida (intento {attempt + 1})")
                        
                    elif response.status == 429:
                        # Rate limit específico con backoff más agresivo
                        self.stats['rate_limited'] += 1
                        # Backoff exponencial más agresivo: base^attempt con multiplicador
                        wait_time = self.config.retry_delay * (2.0 ** attempt) * (1 + random.uniform(0, 0.5))
                        logger.warning(f"🚨 Rate limit! Esperando {wait_time:.1f}s (intento {attempt + 1}/{self.config.retry_attempts})")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        error_text = await response.text()
                        # Backoff exponencial para otros errores HTTP
                        wait_time = self.config.retry_delay * (1.8 ** attempt)
                        logger.warning(f"⚠️ HTTP {response.status}: {error_text[:200]} - Esperando {wait_time:.1f}s (intento {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                
            except asyncio.TimeoutError:
                wait_time = self.config.retry_delay * (1.5 ** attempt)
                logger.warning(f"⏰ Timeout en request - Esperando {wait_time:.1f}s (intento {attempt + 1}/{self.config.retry_attempts})")
                await asyncio.sleep(wait_time)
                continue
                
            except aiohttp.ClientError as e:
                wait_time = self.config.retry_delay * (1.5 ** attempt) * 0.8
                logger.warning(f"🌐 Error de cliente HTTP: {e} - Esperando {wait_time:.1f}s (intento {attempt + 1}/{self.config.retry_attempts})")
                await asyncio.sleep(wait_time)
                continue
                
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "connection" in error_msg:
                    # Backoff exponencial más corto para errores de conexión
                    wait_time = self.config.retry_delay * (1.5 ** attempt) * 0.5
                    logger.warning(f"🔌 Error conexión intento {attempt + 1}/{self.config.retry_attempts}: {e} - Esperando {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    # Backoff exponencial para otros errores
                    wait_time = self.config.retry_delay * (1.8 ** attempt)
                    logger.warning(f"❌ Error intento {attempt + 1}/{self.config.retry_attempts}: {e} - Esperando {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"💥 Falló después de {self.config.retry_attempts} intentos")
        self.stats['failed'] += 1
        return None

class ConservativePrompts:
    """Sistema avanzado de prompts con gran variedad y categorías específicas"""
    
    @staticmethod
    def get_educational_prompts(language: str) -> tuple[str, str]:
        """Prompts educativos y explicativos"""
        if language == "en":
            system_prompts = [
                "You are an expert educator who explains complex topics in simple terms.",
                "You are a knowledgeable teacher who creates engaging educational content.",
                "You are a learning specialist who breaks down difficult concepts clearly.",
                "You are an academic writer who makes information accessible to everyone."
            ]
            
            topics = [
                "the water cycle and its importance for ecosystems",
                "how artificial intelligence is changing modern workplaces",
                "the science behind climate change and global warming",
                "basic principles of economics and personal finance",
                "the human digestive system and nutrition",
                "renewable energy sources and their benefits",
                "the history and impact of the internet",
                "basic chemistry concepts in everyday life",
                "how vaccines work to protect public health",
                "the solar system and space exploration",
                "photosynthesis and plant biology",
                "the importance of biodiversity and conservation"
            ]
            
            topic = random.choice(topics)
            user_prompt = f"Write an educational article explaining {topic}. Use simple language, include examples, and make it engaging for a general audience. Keep it between 300-500 words."
            
        else:
            system_prompts = [
                "Eres un educador experto que explica temas complejos de forma sencilla.",
                "Eres un maestro conocedor que crea contenido educativo atractivo.",
                "Eres un especialista en aprendizaje que desglosa conceptos difíciles claramente.",
                "Eres un escritor académico que hace la información accesible para todos."
            ]
            
            topics = [
                "el ciclo del agua y su importancia para los ecosistemas",
                "cómo la inteligencia artificial está cambiando los lugares de trabajo modernos",
                "la ciencia detrás del cambio climático y el calentamiento global",
                "principios básicos de economía y finanzas personales",
                "el sistema digestivo humano y la nutrición",
                "fuentes de energía renovable y sus beneficios",
                "la historia e impacto del internet",
                "conceptos básicos de química en la vida cotidiana",
                "cómo funcionan las vacunas para proteger la salud pública",
                "el sistema solar y la exploración espacial",
                "la fotosíntesis y la biología vegetal",
                "la importancia de la biodiversidad y la conservación"
            ]
            
            topic = random.choice(topics)
            user_prompt = f"Escribe un artículo educativo explicando {topic}. Usa lenguaje sencillo, incluye ejemplos, y hazlo atractivo para una audiencia general. Manténlo entre 300-500 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_creative_story_prompts(language: str) -> tuple[str, str]:
        """Prompts para historias creativas"""
        if language == "en":
            system_prompts = [
                "You are a creative storyteller who writes engaging narratives with vivid descriptions.",
                "You are a fiction writer who creates compelling characters and interesting plots.",
                "You are a narrative artist who crafts stories that resonate with readers.",
                "You are an imaginative author who brings stories to life with rich details."
            ]
            
            story_elements = [
                ("a young inventor", "discovers an ancient technology", "a hidden underground city"),
                ("a retired teacher", "finds a mysterious letter", "a small coastal town"),
                ("twin siblings", "must solve a family mystery", "their grandmother's old house"),
                ("a street musician", "gains an unexpected opportunity", "a bustling city square"),
                ("a marine biologist", "makes an incredible discovery", "a remote island"),
                ("a young chef", "enters a cooking competition", "a prestigious culinary school"),
                ("an elderly gardener", "unlocks a garden's secret", "a Victorian mansion"),
                ("a librarian", "stumbles upon a hidden room", "an ancient library"),
                ("a park ranger", "encounters something unusual", "a national forest"),
                ("a taxi driver", "picks up an extraordinary passenger", "a rainy night in the city")
            ]
            
            character, plot, setting = random.choice(story_elements)
            user_prompt = f"Write a short story about {character} who {plot} in {setting}. Make it engaging with dialogue, description, and a satisfying conclusion. Keep it between 400-600 words."
            
        else:
            system_prompts = [
                "Eres un narrador creativo que escribe narrativas atractivas con descripciones vivas.",
                "Eres un escritor de ficción que crea personajes convincentes y tramas interesantes.",
                "Eres un artista narrativo que crea historias que resuenan con los lectores.",
                "Eres un autor imaginativo que da vida a las historias con detalles ricos."
            ]
            
            story_elements = [
                ("un joven inventor", "descubre una tecnología antigua", "una ciudad subterránea oculta"),
                ("una maestra jubilada", "encuentra una carta misteriosa", "un pequeño pueblo costero"),
                ("hermanos gemelos", "deben resolver un misterio familiar", "la casa antigua de su abuela"),
                ("un músico callejero", "obtiene una oportunidad inesperada", "una plaza bulliciosa de la ciudad"),
                ("una bióloga marina", "hace un descubrimiento increíble", "una isla remota"),
                ("un joven chef", "participa en una competencia de cocina", "una escuela culinaria prestigiosa"),
                ("un jardinero anciano", "desbloquea el secreto de un jardín", "una mansión victoriana"),
                ("una bibliotecaria", "se topa con una habitación oculta", "una biblioteca antigua"),
                ("un guardaparques", "se encuentra con algo inusual", "un bosque nacional"),
                ("un taxista", "recoge a un pasajero extraordinario", "una noche lluviosa en la ciudad")
            ]
            
            character, plot, setting = random.choice(story_elements)
            user_prompt = f"Escribe una historia corta sobre {character} que {plot} en {setting}. Hazla atractiva con diálogo, descripción y una conclusión satisfactoria. Manténla entre 400-600 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_how_to_prompts(language: str) -> tuple[str, str]:
        """Prompts de instrucciones y guías prácticas"""
        if language == "en":
            system_prompts = [
                "You are a practical guide writer who creates clear, step-by-step instructions.",
                "You are an expert instructor who breaks down complex processes into simple steps.",
                "You are a helpful mentor who provides detailed, actionable guidance.",
                "You are a tutorial specialist who makes learning new skills accessible and easy."
            ]
            
            how_to_topics = [
                "start a small vegetable garden at home",
                "improve your public speaking skills",
                "organize your digital files and photos",
                "create a monthly budget and stick to it",
                "learn a new language effectively",
                "reduce stress through mindfulness techniques",
                "build better relationships with colleagues",
                "develop a consistent exercise routine",
                "improve your sleep quality naturally",
                "cook healthy meals on a tight schedule",
                "start freelancing as a side business",
                "reduce your environmental footprint",
                "prepare for a job interview successfully",
                "organize your living space efficiently",
                "build confidence in social situations"
            ]
            
            topic = random.choice(how_to_topics)
            user_prompt = f"Write a comprehensive guide on how to {topic}. Include specific steps, practical tips, common mistakes to avoid, and helpful resources. Make it actionable and easy to follow. Keep it between 350-500 words."
            
        else:
            system_prompts = [
                "Eres un escritor de guías prácticas que crea instrucciones claras paso a paso.",
                "Eres un instructor experto que desglosa procesos complejos en pasos simples.",
                "Eres un mentor útil que proporciona orientación detallada y procesable.",
                "Eres un especialista en tutoriales que hace que aprender nuevas habilidades sea accesible y fácil."
            ]
            
            how_to_topics = [
                "comenzar un pequeño huerto de vegetales en casa",
                "mejorar tus habilidades para hablar en público",
                "organizar tus archivos digitales y fotos",
                "crear un presupuesto mensual y cumplirlo",
                "aprender un nuevo idioma efectivamente",
                "reducir el estrés a través de técnicas de atención plena",
                "construir mejores relaciones con colegas",
                "desarrollar una rutina de ejercicio consistente",
                "mejorar la calidad del sueño naturalmente",
                "cocinar comidas saludables con horarios ajustados",
                "comenzar a trabajar como freelancer de medio tiempo",
                "reducir tu huella ambiental",
                "prepararse exitosamente para una entrevista de trabajo",
                "organizar tu espacio vital eficientemente",
                "construir confianza en situaciones sociales"
            ]
            
            topic = random.choice(how_to_topics)
            user_prompt = f"Escribe una guía completa sobre cómo {topic}. Incluye pasos específicos, consejos prácticos, errores comunes a evitar y recursos útiles. Hazla procesable y fácil de seguir. Manténla entre 350-500 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_analysis_prompts(language: str) -> tuple[str, str]:
        """Prompts de análisis y opinión"""
        if language == "en":
            system_prompts = [
                "You are a thoughtful analyst who examines topics from multiple perspectives.",
                "You are a critical thinker who provides balanced, well-reasoned analysis.",
                "You are a researcher who presents complex issues in an accessible way.",
                "You are a social commentator who explores important contemporary issues."
            ]
            
            analysis_topics = [
                "the impact of social media on mental health among teenagers",
                "the benefits and challenges of remote work for companies",
                "how streaming services are changing the entertainment industry",
                "the role of artificial intelligence in modern healthcare",
                "the effects of urbanization on community relationships",
                "the importance of digital literacy in today's society",
                "how online shopping is transforming retail businesses",
                "the influence of mobile technology on human behavior",
                "the pros and cons of electric vehicles for the environment",
                "the impact of globalization on local cultures",
                "how video games are being used in education",
                "the changing nature of work in the digital age"
            ]
            
            topic = random.choice(analysis_topics)
            user_prompt = f"Write an analytical piece examining {topic}. Present multiple viewpoints, discuss benefits and challenges, and provide specific examples. Keep your analysis balanced and informative. Aim for 400-550 words."
            
        else:
            system_prompts = [
                "Eres un analista reflexivo que examina temas desde múltiples perspectivas.",
                "Eres un pensador crítico que proporciona análisis equilibrado y bien razonado.",
                "Eres un investigador que presenta temas complejos de manera accesible.",
                "Eres un comentarista social que explora temas contemporáneos importantes."
            ]
            
            analysis_topics = [
                "el impacto de las redes sociales en la salud mental de los adolescentes",
                "los beneficios y desafíos del trabajo remoto para las empresas",
                "cómo los servicios de streaming están cambiando la industria del entretenimiento",
                "el papel de la inteligencia artificial en la atención médica moderna",
                "los efectos de la urbanización en las relaciones comunitarias",
                "la importancia de la alfabetización digital en la sociedad actual",
                "cómo las compras en línea están transformando los negocios minoristas",
                "la influencia de la tecnología móvil en el comportamiento humano",
                "las ventajas y desventajas de los vehículos eléctricos para el medio ambiente",
                "el impacto de la globalización en las culturas locales",
                "cómo los videojuegos se están utilizando en la educación",
                "la naturaleza cambiante del trabajo en la era digital"
            ]
            
            topic = random.choice(analysis_topics)
            user_prompt = f"Escribe un análisis examinando {topic}. Presenta múltiples puntos de vista, discute beneficios y desafíos, y proporciona ejemplos específicos. Mantén tu análisis equilibrado e informativo. Busca entre 400-550 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_dialogue_prompts(language: str) -> tuple[str, str]:
        """Prompts para diálogos y conversaciones"""
        if language == "en":
            system_prompts = [
                "You are a dialogue writer who creates natural, engaging conversations between characters.",
                "You are a scriptwriter who writes realistic dialogue that reveals character personalities.",
                "You are a conversation specialist who crafts meaningful exchanges between people.",
                "You are a dramatic writer who creates compelling dialogue with authentic voices."
            ]
            
            dialogue_scenarios = [
                ("a job interview", "a nervous candidate and an encouraging interviewer"),
                ("a first day at a new school", "a shy student and a friendly classmate"),
                ("a family dinner", "three generations discussing current events"),
                ("a doctor's appointment", "a patient and doctor discussing health concerns"),
                ("a customer service call", "a frustrated customer and a helpful representative"),
                ("a first date", "two people getting to know each other"),
                ("a team meeting", "colleagues brainstorming solutions to a problem"),
                ("a parent-teacher conference", "parents and teacher discussing a student's progress"),
                ("a restaurant scene", "friends catching up over dinner"),
                ("a bookstore encounter", "two strangers bonding over a shared interest in books")
            ]
            
            scenario, characters = random.choice(dialogue_scenarios)
            user_prompt = f"Write a dialogue set during {scenario} between {characters}. Make the conversation natural, include character development, and show personality through speech patterns. Include brief action descriptions. Keep it between 300-450 words."
            
        else:
            system_prompts = [
                "Eres un escritor de diálogos que crea conversaciones naturales y atractivas entre personajes.",
                "Eres un guionista que escribe diálogos realistas que revelan las personalidades de los personajes.",
                "Eres un especialista en conversaciones que crea intercambios significativos entre personas.",
                "Eres un escritor dramático que crea diálogos convincentes con voces auténticas."
            ]
            
            dialogue_scenarios = [
                ("una entrevista de trabajo", "un candidato nervioso y un entrevistador alentador"),
                ("un primer día en una nueva escuela", "un estudiante tímido y un compañero de clase amigable"),
                ("una cena familiar", "tres generaciones discutiendo eventos actuales"),
                ("una cita médica", "un paciente y doctor discutiendo preocupaciones de salud"),
                ("una llamada de servicio al cliente", "un cliente frustrado y un representante útil"),
                ("una primera cita", "dos personas conociéndose"),
                ("una reunión de equipo", "colegas generando ideas para resolver un problema"),
                ("una conferencia de padres y maestros", "padres y maestro discutiendo el progreso de un estudiante"),
                ("una escena en restaurante", "amigos poniéndose al día durante la cena"),
                ("un encuentro en librería", "dos extraños uniéndose por un interés compartido en libros")
            ]
            
            scenario, characters = random.choice(dialogue_scenarios)
            user_prompt = f"Escribe un diálogo ambientado durante {scenario} entre {characters}. Haz la conversación natural, incluye desarrollo de personajes y muestra personalidad a través de patrones de habla. Incluye breves descripciones de acción. Manténlo entre 300-450 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_technical_prompts(language: str) -> tuple[str, str]:
        """Prompts técnicos y de programación"""
        if language == "en":
            system_prompts = [
                "You are a technical writer who explains programming concepts clearly and practically.",
                "You are a software developer who creates educational content about coding.",
                "You are a technology instructor who makes complex technical topics accessible.",
                "You are a programming mentor who provides clear explanations and examples."
            ]
            
            tech_topics = [
                ("Python", "create a simple web scraper"),
                ("JavaScript", "build a todo list application"),
                ("HTML/CSS", "design a responsive navigation menu"),
                ("Python", "implement a basic data analysis script"),
                ("JavaScript", "create an interactive quiz application"),
                ("SQL", "design a database for a small business"),
                ("Python", "build a password generator tool"),
                ("JavaScript", "create a weather dashboard"),
                ("CSS", "design a modern card layout"),
                ("Python", "create a file organizer script")
            ]
            
            language_tech, project = random.choice(tech_topics)
            user_prompt = f"Write a technical tutorial on how to {project} using {language_tech}. Include code examples, step-by-step instructions, and explanations of key concepts. Make it beginner-friendly but comprehensive. Aim for 400-600 words."
            
        else:
            system_prompts = [
                "Eres un escritor técnico que explica conceptos de programación de forma clara y práctica.",
                "Eres un desarrollador de software que crea contenido educativo sobre codificación.",
                "Eres un instructor de tecnología que hace accesibles temas técnicos complejos.",
                "Eres un mentor de programación que proporciona explicaciones claras y ejemplos."
            ]
            
            tech_topics = [
                ("Python", "crear un web scraper simple"),
                ("JavaScript", "construir una aplicación de lista de tareas"),
                ("HTML/CSS", "diseñar un menú de navegación responsivo"),
                ("Python", "implementar un script básico de análisis de datos"),
                ("JavaScript", "crear una aplicación de quiz interactivo"),
                ("SQL", "diseñar una base de datos para un pequeño negocio"),
                ("Python", "construir una herramienta generadora de contraseñas"),
                ("JavaScript", "crear un dashboard del clima"),
                ("CSS", "diseñar un layout moderno de tarjetas"),
                ("Python", "crear un script organizador de archivos")
            ]
            
            language_tech, project = random.choice(tech_topics)
            user_prompt = f"Escribe un tutorial técnico sobre cómo {project} usando {language_tech}. Incluye ejemplos de código, instrucciones paso a paso y explicaciones de conceptos clave. Hazlo amigable para principiantes pero completo. Busca entre 400-600 palabras."
        
        return random.choice(system_prompts), user_prompt
    
    @staticmethod
    def get_random_prompt(language: str = "es") -> tuple[str, str]:
        """Selección aleatoria de cualquier categoría de prompt"""
        prompt_categories = [
            ConservativePrompts.get_educational_prompts,
            ConservativePrompts.get_creative_story_prompts,
            ConservativePrompts.get_how_to_prompts,
            ConservativePrompts.get_analysis_prompts,
            ConservativePrompts.get_dialogue_prompts,
            ConservativePrompts.get_technical_prompts
        ]
        
        # Distribución ponderada para más variedad
        weights = [20, 18, 20, 15, 12, 15]  # Porcentajes aproximados
        selected_category = random.choices(prompt_categories, weights=weights)[0]
        
        return selected_category(language)

class CheckpointDatasetGenerator:
    """Generador con sistema de checkpoints robusto"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self.generated_count = 0
        self.current_batch_id = 0
    
    async def generate_single_example(self, client: ConservativeAzureClient, example_id: int) -> Optional[Dict[str, Any]]:
        """Generar un ejemplo con máximo cuidado"""
        current_lang = self.config.language
        if current_lang == "mixed":
            current_lang = random.choice(["es", "en"])
        
        system_prompt, user_prompt = ConservativePrompts.get_random_prompt(current_lang)
        
        result = await client.generate_safely(system_prompt, user_prompt)
        
        if result:
            return {
                "text": result,
                "id": example_id,
                "timestamp": time.time(),
                "language": current_lang
            }
        
        return None
    
    async def generate_batch(self, client: ConservativeAzureClient, batch_id: int) -> List[Dict[str, Any]]:
        """Generar lote con máxima concurrencia interna"""
        # Crear semáforo para controlar la concurrencia interna del batch
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def generate_with_semaphore(example_idx):
            async with semaphore:
                # Generar ID único basado en batch_id y example_idx
                global_example_id = batch_id * self.config.batch_size + example_idx
                return await self.generate_single_example(client, global_example_id)
        
        # Generar todos los ejemplos del batch concurrentemente
        tasks = [generate_with_semaphore(i) for i in range(self.config.batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados válidos
        batch_results = []
        for result in results:
            if isinstance(result, dict) and result is not None:
                batch_results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"⚠️ Error en ejemplo del batch: {result}")
        
        # Delay mínimo entre lotes
        if batch_results:
            await asyncio.sleep(self.config.batch_delay)
        
        return batch_results
    
    def save_batch(self, batch_data: List[Dict[str, Any]], batch_id: int):
        """Guardar lote con checkpoints automáticos"""
        if not batch_data:
            return
        
        filename = self.output_dir / f"batch_{batch_id:06d}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.generated_count += len(batch_data)
        self.current_batch_id = batch_id
        
        progress_pct = (self.generated_count / self.config.target_size) * 100
        remaining_time = self.checkpoint_manager.estimate_remaining_time(self.generated_count, self.config.target_size)
        
        logger.info(f"💾 Lote {batch_id + 1} guardado: {len(batch_data)} elementos")
        logger.info(f"📊 Total: {self.generated_count:,}/{self.config.target_size:,} ({progress_pct:.1f}%)")
        logger.info(f"⏰ Tiempo restante: {remaining_time:.1f}h")
        
        # Sistema de checkpoints por intervalos
        if self.generated_count % self.config.checkpoint_interval == 0:
            # Checkpoint principal cada 1000
            self.checkpoint_manager.save_checkpoint(
                self.generated_count, 
                batch_id, 
                self.config, 
                {"reason": "milestone", "batch_size": len(batch_data)}
            )
        elif self.generated_count % self.config.mini_checkpoint_interval == 0:
            # Mini-checkpoint cada 100
            self.checkpoint_manager.save_progress(
                self.generated_count, 
                batch_id, 
                {"recent_batch_size": len(batch_data)}
            )
        elif self.generated_count % self.config.progress_save_interval == 0:
            # Progress cada 50
            self.checkpoint_manager.save_progress(self.generated_count, batch_id)
    
    async def generate_with_checkpoints(self):
        """Generación principal con sistema de checkpoints"""
        logger.info(f"🚀 Iniciando generación AGRESIVA con CHECKPOINTS ROBUSTOS")
        logger.info(f"🎯 Target: {self.config.target_size:,} ejemplos")
        logger.info(f"💾 Checkpoints cada {self.config.checkpoint_interval} ejemplos")
        logger.info(f"⚡ MODO AGRESIVO: {self.config.max_concurrent} concurrent, delay {self.config.request_delay}s")
        logger.info(f"📦 Batch size: {self.config.batch_size}, batch delay: {self.config.batch_delay}s")
        
        # Verificar API key
        api_key = os.getenv("AZURE_AI_API_KEY")
        if not api_key:
            logger.error("❌ Variable AZURE_AI_API_KEY no encontrada")
            return
        
        # Cargar checkpoint
        start_count, start_batch, _ = self.checkpoint_manager.load_checkpoint()
        self.generated_count = start_count
        
        total_batches = (self.config.target_size + self.config.batch_size - 1) // self.config.batch_size
        
        if start_count > 0:
            logger.info(f"🔄 REANUDANDO desde {start_count:,} ejemplos (lote {start_batch + 1})")
        else:
            logger.info(f"🆕 Comenzando desde cero")
            # Checkpoint inicial
            self.checkpoint_manager.save_checkpoint(0, 0, self.config, {"reason": "start"})
        
        async with ConservativeAzureClient(self.config.azure_endpoint, api_key, self.config.model_name, self.config) as client:
            # Test de conexión con grok-3-mini (formato chat)
            logger.info("🧪 Probando conexión con grok-3-mini (formato chat)...")
            test_result = await client.generate_safely("Eres un asistente útil.", "Di hola en español.")
            if not test_result:
                logger.error("❌ Test de conexión con grok-3-mini falló")
                return
            
            logger.info("✅ Conexión con grok-3-mini exitosa (formato chat)")
            logger.info(f"🔍 Test response: {test_result[:100]}...")
            
            # Procesar lotes
            for batch_id in range(start_batch, total_batches):
                if self.generated_count >= self.config.target_size:
                    break
                
                logger.info(f"\n🎯 === LOTE {batch_id + 1}/{total_batches} ===")
                
                try:
                    batch_data = await self.generate_batch(client, batch_id)
                    
                    if batch_data:
                        self.save_batch(batch_data, batch_id)
                        
                        # Recovery info después de cada lote exitoso
                        self.checkpoint_manager.save_recovery_info(
                            self.generated_count, 
                            batch_id, 
                            f"Lote {batch_id + 1} completado exitosamente"
                        )
                    else:
                        logger.warning(f"⚠️ Lote {batch_id + 1} vacío")
                        
                except Exception as e:
                    logger.error(f"❌ Error en lote {batch_id + 1}: {e}")
                    self.checkpoint_manager.save_recovery_info(
                        self.generated_count, 
                        batch_id, 
                        f"Error en lote {batch_id + 1}: {str(e)}"
                    )
                    raise
        
        # Checkpoint final
        logger.info(f"💾 Guardando checkpoint final...")
        self.checkpoint_manager.save_checkpoint(
            self.generated_count, 
            total_batches - 1, 
            self.config, 
            {"reason": "completion"}
        )
        
        logger.info(f"🎉 Generación completada: {self.generated_count:,} ejemplos")
        
        # Limpiar recovery file (ya no necesario)
        if self.checkpoint_manager.recovery_file.exists():
            self.checkpoint_manager.recovery_file.unlink()
        
        # Consolidar dataset
        self.consolidate_dataset()
    
    def consolidate_dataset(self):
        """Consolidar todos los archivos"""
        logger.info("📋 Consolidando dataset...")
        
        output_file = self.output_dir / "complete_checkpoint_dataset.jsonl"
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        if not batch_files:
            logger.warning("⚠️ No hay archivos para consolidar")
            return
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for batch_file in batch_files:
                with open(batch_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
        
        # Stats
        total_lines = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
        file_size = output_file.stat().st_size / (1024**2)
        
        logger.info(f"✅ Dataset consolidado:")
        logger.info(f"  📄 Archivo: {output_file}")
        logger.info(f"  📊 Elementos: {total_lines:,}")
        logger.info(f"  💾 Tamaño: {file_size:.2f} MB")

def show_checkpoint_status(output_dir: str):
    """Mostrar estado de checkpoints"""
    output_path = Path(output_dir)
    checkpoint_file = output_path / "checkpoint.json"
    recovery_file = output_path / "recovery.json"
    
    if not checkpoint_file.exists() and not recovery_file.exists():
        return
    
    print(f"\n🔍 ESTADO DE CHECKPOINTS en '{output_dir}':")
    print("=" * 60)
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            print(f"💾 CHECKPOINT Principal:")
            print(f"   📊 Ejemplos: {data.get('generated_count', 0):,}")
            print(f"   🎯 Progreso: {data.get('progress_pct', 0):.1f}%")
            print(f"   📅 Fecha: {time.ctime(data.get('timestamp', 0))}")
            print(f"   ⏰ Tiempo restante: {data.get('estimated_remaining_hours', 0):.1f}h")
            print(f"   🔧 Razón: {data.get('checkpoint_reason', 'N/A')}")
            
            if 'session_stats' in data:
                stats = data['session_stats']
                print(f"   📈 Velocidad: {stats.get('avg_examples_per_hour', 0):.1f} ejemplos/hora")
                
        except Exception as e:
            print(f"❌ Error leyendo checkpoint: {e}")
    
    if recovery_file.exists():
        try:
            with open(recovery_file, 'r') as f:
                data = json.load(f)
            
            print(f"\n🛠️ RECOVERY INFO:")
            print(f"   📊 Último exitoso: {data.get('last_successful_count', 0):,}")
            print(f"   ❗ Último error: {data.get('error_info', 'N/A')}")
            print(f"   🔄 Reanudar desde lote: {data.get('recovery_instructions', {}).get('resume_from_batch', 0)}")
            
        except Exception as e:
            print(f"❌ Error leyendo recovery: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generador con CHECKPOINTS ROBUSTOS para Azure AI")
    parser.add_argument("--size", type=int, default=1000, help="Tamaño objetivo")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamaño del lote")
    parser.add_argument("--concurrent", type=int, default=50, help="Concurrencia (max 100)")
    parser.add_argument("--output", type=str, default="checkpoint_dataset", help="Directorio")
    parser.add_argument("--model", type=str, default="grok-3-mini", help="Modelo")
    parser.add_argument("--language", type=str, default="es", choices=["es", "en", "mixed"], help="Idioma")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Intervalo checkpoint")
    parser.add_argument("--request-delay", type=float, default=0.05, help="Delay requests")
    parser.add_argument("--batch-delay", type=float, default=0.1, help="Delay lotes")
    parser.add_argument("--consolidate-only", action="store_true", help="Solo consolidar")
    parser.add_argument("--status", action="store_true", help="Ver estado checkpoints")
    parser.add_argument("--clean-checkpoints", action="store_true", help="Limpiar checkpoints")
    
    args = parser.parse_args()
    
    # Mostrar estado
    if args.status:
        show_checkpoint_status(args.output)
        return
    
    # Limpiar checkpoints
    if args.clean_checkpoints:
        output_path = Path(args.output)
        files_to_clean = ["checkpoint.json", "progress.json", "recovery.json"]
        for filename in files_to_clean:
            file_path = output_path / filename
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️ Eliminado: {filename}")
        print("✅ Checkpoints limpiados")
        return
    
    # Validaciones menos restrictivas para modo agresivo
    if args.concurrent > 100:
        logger.warning("⚠️ Concurrencia máxima recomendada: 100")
        args.concurrent = 100
    
    config = CheckpointConfig(
        target_size=args.size,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent,
        output_dir=args.output,
        model_name=args.model,
        language=args.language,
        checkpoint_interval=args.checkpoint_interval,
        request_delay=args.request_delay,
        batch_delay=args.batch_delay
    )
    
    generator = CheckpointDatasetGenerator(config)
    
    if args.consolidate_only:
        generator.consolidate_dataset()
        return
    
    # Mostrar estado previo si existe
    show_checkpoint_status(args.output)
    
    logger.info(f"🚀 Iniciando generación con checkpoints cada {args.checkpoint_interval} ejemplos")
    logger.info(f"💡 Puedes interrumpir (Ctrl+C) y reanudar en cualquier momento")
    
    try:
        asyncio.run(generator.generate_with_checkpoints())
    except KeyboardInterrupt:
        logger.info(f"\n⏸️ Generación pausada por usuario")
        generator.checkpoint_manager.save_recovery_info(
            generator.generated_count, 
            generator.current_batch_id, 
            "Pausa manual por usuario (Ctrl+C)"
        )
        logger.info(f"💾 Progreso guardado. Usa el mismo comando para reanudar.")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        generator.checkpoint_manager.save_recovery_info(
            generator.generated_count, 
            generator.current_batch_id, 
            f"Error crítico: {str(e)}"
        )
        raise

if __name__ == "__main__":
    main()