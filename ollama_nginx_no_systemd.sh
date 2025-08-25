#!/bin/bash

# Script dinámico para Ollama con distribución inteligente de GPUs
# Detecta GPUs y asigna según VRAM disponible para nemotron (43GB)

# set -e  # Comentado para debug - ver errores completos

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables globales
MODEL_NAME="nemotron"
MODEL_VRAM_REQUIRED=43  # GB requeridos por nemotron
NGINX_CONF_PATH="/etc/nginx/sites-available/ollama-lb"
NGINX_ENABLED_PATH="/etc/nginx/sites-enabled/ollama-lb"
OLLAMA_LOG_DIR="/var/log/ollama"
BASE_PORT=11434

# Arrays para almacenar información de GPUs e instancias
declare -a GPU_MEMORY=()
declare -a GPU_IDS=()
declare -a INSTANCE_CONFIGS=()

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# Detectar y analizar GPUs
detect_gpus() {
    print_status "Detectando GPUs disponibles..."
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        print_error "nvidia-smi no encontrado. Instala los drivers NVIDIA"
        exit 1
    fi
    
    # Obtener información de GPUs
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
    
    if [ -z "$gpu_info" ]; then
        print_error "No se encontraron GPUs compatibles"
        exit 1
    fi
    
    print_status "GPUs detectadas:"
    echo "$gpu_info" | while IFS=, read -r index name memory_mb; do
        # Convertir MB a GB (redondeado hacia abajo)
        local memory_gb=$((memory_mb / 1024))
        GPU_IDS+=($index)
        GPU_MEMORY+=($memory_gb)
        
        printf "  GPU %d: %s - %d GB VRAM\n" "$index" "$(echo $name | xargs)" "$memory_gb"
    done
    
    # Recargar arrays (necesario debido al pipe)
    GPU_IDS=()
    GPU_MEMORY=()
    while IFS=, read -r index name memory_mb; do
        local memory_gb=$((memory_mb / 1024))
        GPU_IDS+=($index)
        GPU_MEMORY+=($memory_gb)
    done <<< "$gpu_info"
    
    local total_gpus=${#GPU_IDS[@]}
    print_status "Total GPUs detectadas: $total_gpus"
}

# Calcular distribución óptima de GPUs
calculate_gpu_distribution() {
    print_status "Calculando distribución óptima para modelo $MODEL_NAME (${MODEL_VRAM_REQUIRED}GB)..."
    
    local total_gpus=${#GPU_IDS[@]}
    local instance_counter=0
    local i=0
    
    while [ $i -lt $total_gpus ]; do
        local current_gpu=${GPU_IDS[$i]}
        local current_memory=${GPU_MEMORY[$i]}
        local gpu_group=($current_gpu)
        local total_memory=$current_memory
        
        print_debug "Evaluando GPU $current_gpu con ${current_memory}GB..."
        
        # Si una GPU tiene suficiente memoria, úsala sola
        if [ $current_memory -ge $MODEL_VRAM_REQUIRED ]; then
            print_status "✅ GPU $current_gpu tiene suficiente VRAM (${current_memory}GB >= ${MODEL_VRAM_REQUIRED}GB)"
            
            local port=$((BASE_PORT + instance_counter))
            local config="INSTANCE_${instance_counter}:PORT=${port}:GPUS=${current_gpu}:MEMORY=${current_memory}GB"
            INSTANCE_CONFIGS+=("$config")
            
            ((instance_counter++))
            ((i++))
            continue
        fi
        
        # Si no tiene suficiente, combinar con GPUs siguientes
        print_debug "GPU $current_gpu insuficiente (${current_memory}GB), buscando combinación..."
        
        local j=$((i + 1))
        while [ $j -lt $total_gpus ] && [ $total_memory -lt $MODEL_VRAM_REQUIRED ]; do
            local next_gpu=${GPU_IDS[$j]}
            local next_memory=${GPU_MEMORY[$j]}
            
            gpu_group+=($next_gpu)
            total_memory=$((total_memory + next_memory))
            
            print_debug "Agregando GPU $next_gpu (+${next_memory}GB), total: ${total_memory}GB"
            
            ((j++))
        done
        
        # Verificar si la combinación es suficiente
        if [ $total_memory -ge $MODEL_VRAM_REQUIRED ]; then
            local gpu_list=$(IFS=,; echo "${gpu_group[*]}")
            local port=$((BASE_PORT + instance_counter))
            local config="INSTANCE_${instance_counter}:PORT=${port}:GPUS=${gpu_list}:MEMORY=${total_memory}GB"
            INSTANCE_CONFIGS+=("$config")
            
            print_status "✅ Instancia $instance_counter: GPUs [${gpu_list}] = ${total_memory}GB"
            
            ((instance_counter++))
            i=$j
        else
            print_warning "❌ GPUs restantes insuficientes para otra instancia (${total_memory}GB < ${MODEL_VRAM_REQUIRED}GB)"
            break
        fi
    done
    
    local total_instances=${#INSTANCE_CONFIGS[@]}
    
    if [ $total_instances -eq 0 ]; then
        print_error "No se puede crear ninguna instancia. VRAM total insuficiente."
        print_error "Requerido: ${MODEL_VRAM_REQUIRED}GB por instancia"
        exit 1
    fi
    
    print_status "🚀 Distribución final: $total_instances instancias"
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance=$(echo $config | cut -d: -f1)
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        local memory=$(echo $config | cut -d: -f4 | cut -d= -f2)
        
        printf "  %s -> Puerto %s, GPUs [%s], VRAM: %s\n" "$instance" "$port" "$gpus" "$memory"
    done
}

# Instalar dependencias
install_dependencies() {
    print_status "Instalando dependencias..."
    
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y curl wget nginx procps net-tools screen supervisor bc
    elif command -v yum >/dev/null 2>&1; then
        yum update -y
        yum install -y curl wget nginx procps net-tools screen supervisor bc
    elif command -v apk >/dev/null 2>&1; then
        apk update
        apk add curl wget nginx procps net-tools screen supervisor bc
    else
        print_error "Gestor de paquetes no soportado"
        exit 1
    fi
}

# Instalar Ollama
install_ollama() {
    print_status "Verificando instalación de Ollama..."
    
    if ! command -v ollama >/dev/null 2>&1; then
        print_status "Ollama no encontrado, instalando..."
        
        # Descargar e instalar Ollama
        curl -fsSL https://ollama.com/install.sh | sh
        
        if [ $? -ne 0 ]; then
            print_error "Error al instalar Ollama"
            exit 1
        fi
        
        # Verificar instalación
        if command -v ollama >/dev/null 2>&1; then
            print_status "✅ Ollama instalado correctamente"
            ollama --version
        else
            print_error "Ollama no se instaló correctamente"
            exit 1
        fi
        
        # Crear usuario ollama si no existe
        if ! id "ollama" &>/dev/null; then
            print_status "Creando usuario ollama..."
            useradd -r -s /bin/false -d /usr/share/ollama ollama
        fi
    else
        print_status "✅ Ollama ya está instalado"
        ollama --version
    fi
    
    # Verificar que Ollama funciona
    print_status "Verificando funcionamiento de Ollama..."
    if ollama list >/dev/null 2>&1; then
        print_status "✅ Ollama funciona correctamente"
    else
        print_warning "Ollama instalado pero puede requerir configuración adicional"
    fi
}

# Crear directorios
create_directories() {
    print_status "Creando estructura de directorios..."
    mkdir -p "$OLLAMA_LOG_DIR"
    mkdir -p /usr/share/ollama/.ollama/models
    chown -R ollama:ollama /usr/share/ollama 2>/dev/null || true
    chown -R ollama:ollama "$OLLAMA_LOG_DIR" 2>/dev/null || true
}

# Generar configuración nginx dinámica
configure_nginx() {
    print_status "Configurando nginx load balancer..."
    
    local total_instances=${#INSTANCE_CONFIGS[@]}
    
    # Crear directorio
    mkdir -p /etc/nginx/sites-available
    mkdir -p /etc/nginx/sites-enabled
    
    # Generar upstream dinámico
    cat > "$NGINX_CONF_PATH" << EOF
upstream ollama_backend {
    least_conn;
EOF
    
    # Agregar cada instancia al upstream
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        echo "    server 127.0.0.1:$port max_fails=3 fail_timeout=30s;" >> "$NGINX_CONF_PATH"
    done
    
    # Completar configuración
    cat >> "$NGINX_CONF_PATH" << 'EOF'
}

server {
    listen 80;
    listen [::]:80;
    
    server_name localhost ollama.local _;
    
    # Configuración para modelos grandes
    client_max_body_size 15G;
    client_body_timeout 300s;
    client_header_timeout 300s;
    
    # Logs
    access_log /var/log/nginx/ollama-access.log;
    error_log /var/log/nginx/ollama-error.log;
    
    # Página de estado personalizada
    location /status {
        access_log off;
        return 200 "Ollama Load Balancer - GPUs: $upstream_addr\n";
        add_header Content-Type text/plain;
    }
    
    # Endpoint principal
    location / {
        proxy_pass http://ollama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts para generación de texto larga
        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Sin buffering para streaming
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF
    
    # Activar configuración
    ln -sf "$NGINX_CONF_PATH" "$NGINX_ENABLED_PATH"
    rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
    
    # Verificar
    nginx -t
    print_status "Nginx configurado para $total_instances instancias"
}

# Configurar supervisor dinámicamente
configure_supervisor() {
    print_status "Configurando supervisor..."
    
    # Crear configuración base
    cat > /etc/supervisor/conf.d/ollama.conf << 'EOF'
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700

[supervisord]
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor/

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

EOF
    
    # Agregar cada instancia
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance=$(echo $config | cut -d: -f1)
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        local instance_num=$(echo $instance | sed 's/INSTANCE_//')
        
        cat >> /etc/supervisor/conf.d/ollama.conf << EOF

[program:ollama-$instance_num]
command=/usr/local/bin/ollama serve
environment=OLLAMA_HOST="0.0.0.0:$port",CUDA_VISIBLE_DEVICES="$gpus",OLLAMA_MODELS="/usr/share/ollama/.ollama/models",OLLAMA_NUM_PARALLEL="1"
user=ollama
directory=/usr/share/ollama
autostart=true
autorestart=true
startsecs=10
startretries=3
redirect_stderr=true
stdout_logfile=$OLLAMA_LOG_DIR/instance-$instance_num.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=$OLLAMA_LOG_DIR/instance-$instance_num-error.log
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
EOF
    done
    
    # Crear grupo
    echo "" >> /etc/supervisor/conf.d/ollama.conf
    echo "[group:ollama]" >> /etc/supervisor/conf.d/ollama.conf
    echo -n "programs=" >> /etc/supervisor/conf.d/ollama.conf
    
    local programs=()
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        programs+=("ollama-$instance_num")
    done
    
    IFS=,; echo "${programs[*]}" >> /etc/supervisor/conf.d/ollama.conf
    echo "priority=999" >> /etc/supervisor/conf.d/ollama.conf
}

# Método alternativo con screen
start_with_screen() {
    print_status "Iniciando con screen sessions..."
    
    # Limpiar sesiones existentes
    screen -ls | grep ollama | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null || true
    
    # Crear sesión para cada instancia
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        
        print_status "Iniciando instancia $instance_num (puerto $port, GPUs: $gpus)"
        
        screen -dmS "ollama-$instance_num" bash -c "
            export OLLAMA_HOST=0.0.0.0:$port
            export CUDA_VISIBLE_DEVICES=$gpus
            export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
            export OLLAMA_NUM_PARALLEL=1
            cd /usr/share/ollama
            sudo -u ollama /usr/local/bin/ollama serve 2>&1 | tee $OLLAMA_LOG_DIR/instance-$instance_num.log
        "
        
        sleep 2
    done
}

# Iniciar servicios
start_services() {
    print_status "Iniciando servicios..."
    
    if command -v supervisorctl >/dev/null 2>&1; then
        # Usar supervisor
        configure_supervisor
        supervisord -c /etc/supervisor/supervisord.conf 2>/dev/null || true
        sleep 3
        supervisorctl reread
        supervisorctl update
        supervisorctl start ollama:*
    else
        # Usar screen
        start_with_screen
    fi
    
    # Iniciar nginx
    service nginx restart 2>/dev/null || /etc/init.d/nginx restart
    
    sleep 5
    print_status "Servicios iniciados"
}

# Verificar servicios
verify_services() {
    print_status "Verificando servicios..."
    
    # Verificar puertos
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        
        if netstat -tuln | grep -q ":$port "; then
            print_status "✅ Instancia $instance_num (puerto $port, GPUs: $gpus): ACTIVA"
        else
            print_warning "❌ Instancia $instance_num (puerto $port): INACTIVA"
        fi
    done
    
    # Verificar nginx
    if netstat -tuln | grep -q ":80 "; then
        print_status "✅ Load Balancer (puerto 80): ACTIVO"
    else
        print_warning "❌ Load Balancer: INACTIVO"
    fi
    
    # Test de conectividad API
    sleep 3
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        
        if curl -s --connect-timeout 5 http://127.0.0.1:$port/api/tags >/dev/null 2>&1; then
            print_status "✅ API instancia $instance_num: FUNCIONANDO"
        else
            print_warning "❌ API instancia $instance_num: NO RESPONDE"
        fi
    done
}

# Descargar modelo
download_model() {
    print_status "Descargando modelo $MODEL_NAME..."
    
    # Esperar que al menos una instancia esté lista
    local ready=false
    for i in {1..60}; do
        for config in "${INSTANCE_CONFIGS[@]}"; do
            local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
            if curl -s --connect-timeout 3 http://127.0.0.1:$port/api/tags >/dev/null 2>&1; then
                print_status "✅ Instancia en puerto $port lista para descarga"
                OLLAMA_HOST=127.0.0.1:$port /usr/local/bin/ollama pull $MODEL_NAME
                ready=true
                break
            fi
        done
        
        if [ "$ready" = true ]; then
            break
        fi
        
        print_status "Esperando instancias... ($i/60)"
        sleep 3
    done
    
    if [ "$ready" = false ]; then
        print_error "Ninguna instancia respondió para descargar el modelo"
        return 1
    fi
    
    # Verificar descarga
    local first_port=$(echo ${INSTANCE_CONFIGS[0]} | cut -d: -f2 | cut -d= -f2)
    if OLLAMA_HOST=127.0.0.1:$first_port /usr/local/bin/ollama list | grep -q $MODEL_NAME; then
        print_status "✅ Modelo $MODEL_NAME disponible"
    else
        print_warning "❌ No se pudo verificar el modelo"
    fi
}

# Crear scripts de utilidad
create_utility_scripts() {
    print_status "Creando scripts de monitoreo..."
    
    # Script de monitoreo avanzado
    cat > /usr/local/bin/ollama-monitor.sh << 'EOF'
#!/bin/bash

echo "🚀 === Monitor Ollama GPU Load Balancer ==="
echo

# Información de GPUs
echo "💾 Estado GPUs:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name util_gpu util_mem mem_used mem_total temp; do
        printf "  GPU %s: %s\n" "$idx" "$(echo $name | xargs)"
        printf "    Uso GPU: %s%% | Mem: %s%% (%s/%s MB) | Temp: %s°C\n" "$util_gpu" "$util_mem" "$mem_used" "$mem_total" "$temp"
    done
else
    echo "  nvidia-smi no disponible"
fi

echo
echo "🔧 Servicios:"
EOF
    
    # Agregar verificación dinámica de instancias
    local instance_checks=""
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        
        instance_checks+="""
# Instancia $instance_num
if curl -s --connect-timeout 3 http://127.0.0.1:$port/api/tags >/dev/null 2>&1; then
    echo \"  ✅ Instancia $instance_num (Puerto $port, GPUs: $gpus): OK\"
else
    echo \"  ❌ Instancia $instance_num (Puerto $port, GPUs: $gpus): ERROR\"
fi
"""
    done
    
    cat >> /usr/local/bin/ollama-monitor.sh << EOF
$instance_checks

# Load Balancer
if curl -s --connect-timeout 3 http://127.0.0.1/api/tags >/dev/null 2>&1; then
    echo "  ✅ Load Balancer (puerto 80): OK"
else
    echo "  ❌ Load Balancer (puerto 80): ERROR"
fi

echo
echo "📊 Procesos activos:"
ps aux | grep "[o]llama serve" | while read -r line; do
    echo "  \$line"
done

echo
echo "📺 Sesiones Screen:"
screen -list 2>/dev/null | grep ollama || echo "  No hay sesiones screen"

echo
echo "🌐 Puertos en uso:"
netstat -tuln | grep -E ":($(echo "${INSTANCE_CONFIGS[@]}" | grep -o 'PORT=[0-9]*' | cut -d= -f2 | tr '\n' '|' | sed 's/|$//')|80)\\s" || echo "  Ningún puerto activo"
EOF

    # Script de reinicio
    cat > /usr/local/bin/ollama-restart.sh << 'EOF'
#!/bin/bash

echo "🔄 Reiniciando Ollama Load Balancer..."

# Detener procesos
pkill -f "ollama serve" 2>/dev/null || true
screen -ls | grep ollama | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null || true

sleep 3

# Reiniciar con supervisor o screen
if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl restart ollama:*
    echo "✅ Reiniciado con supervisor"
else
EOF
    
    # Agregar reinicio screen dinámico
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        
        cat >> /usr/local/bin/ollama-restart.sh << EOF
    screen -dmS "ollama-$instance_num" bash -c "
        export OLLAMA_HOST=0.0.0.0:$port
        export CUDA_VISIBLE_DEVICES=$gpus
        export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
        export OLLAMA_NUM_PARALLEL=1
        cd /usr/share/ollama
        sudo -u ollama /usr/local/bin/ollama serve 2>&1 | tee $OLLAMA_LOG_DIR/instance-$instance_num.log
    "
EOF
    done
    
    cat >> /usr/local/bin/ollama-restart.sh << 'EOF'
    echo "✅ Reiniciado con screen"
fi

# Reiniciar nginx
service nginx restart 2>/dev/null || /etc/init.d/nginx restart
echo "✅ Nginx reiniciado"

sleep 5
/usr/local/bin/ollama-monitor.sh
EOF

    # Script de test con distribución de carga
    cat > /usr/local/bin/test-ollama.sh << EOF
#!/bin/bash

echo "🧪 === Test Ollama Load Balancer ==="
echo

echo "Modelo: $MODEL_NAME"
echo "Instancias activas: ${#INSTANCE_CONFIGS[@]}"
echo

# Test básico
echo "📝 Test básico:"
curl -X POST http://localhost/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "$MODEL_NAME",
    "prompt": "¿Qué es deep learning? Responde en 2 líneas.",
    "stream": false
  }' 2>/dev/null | jq -r '.response' 2>/dev/null || echo "Error en conexión"

echo
echo "⚡ Test de carga (3 requests simultáneos):"

# Test de carga
for i in {1..3}; do
    (
        echo "Request \$i iniciado..."
        curl -X POST http://localhost/api/generate \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "$MODEL_NAME",
            "prompt": "Cuenta hasta 5 y explica brevemente qué es la IA. Request \$i",
            "stream": false
          }' 2>/dev/null | jq -r '.response' | head -2
        echo "Request \$i completado"
    ) &
done

wait
echo "✅ Test de carga completado"
EOF

    chmod +x /usr/local/bin/ollama-monitor.sh
    chmod +x /usr/local/bin/ollama-restart.sh  
    chmod +x /usr/local/bin/test-ollama.sh
}

# Función principal
main() {
    if [ "$EUID" -ne 0 ]; then
        print_error "Ejecutar como root: sudo $0"
        exit 1
    fi
    
    echo -e "${GREEN}🚀 === Ollama GPU Load Balancer Dinámico ===${NC}"
    echo -e "${GREEN}Modelo: $MODEL_NAME (${MODEL_VRAM_REQUIRED}GB VRAM requerida)${NC}"
    echo
    
    detect_gpus
    calculate_gpu_distribution
    
    echo
    read -p "¿Continuar con esta configuración? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Instalación cancelada"
        exit 0
    fi
    
    install_dependencies
    install_ollama  
    create_directories
    configure_nginx
    start_services
    verify_services
    download_model
    create_utility_scripts
    
    echo
    print_status "🎉 ¡Instalación completada!"
    echo -e "${GREEN}=== RESUMEN ===${NC}"
    echo "🎯 Modelo: $MODEL_NAME"
    echo "🔧 Instancias: ${#INSTANCE_CONFIGS[@]}"
    
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_num=$(echo $config | cut -d: -f1 | sed 's/INSTANCE_//')
        local port=$(echo $config | cut -d: -f2 | cut -d= -f2)
        local gpus=$(echo $config | cut -d: -f3 | cut -d= -f2)
        local memory=$(echo $config | cut -d: -f4 | cut -d= -f2)
        echo "  • Instancia $instance_num: Puerto $port, GPUs [$gpus], VRAM: $memory"
    done
    
    echo
    echo "🌐 Load Balancer: http://localhost"
    echo "📊 Monitor: /usr/local/bin/ollama-monitor.sh"
    echo "🔄 Reiniciar: /usr/local/bin/ollama-restart.sh"
    echo "🧪 Probar: /usr/local/bin/test-ollama.sh"
    echo
    
    # Mostrar estado final
    /usr/local/bin/ollama-monitor.sh
}

# Ejecutar
main "$@"