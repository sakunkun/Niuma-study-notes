#!/bin/bash

# 分布式训练脚本 - 优化版本
# 作者: 优化版本
# 功能: 在多个节点上启动分布式深度学习训练

set -euo pipefail  # 严格模式：遇到错误立即退出

# 全局变量
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG_FILE="${SCRIPT_DIR}/distribute_config.env"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO $(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/distribute_${TIMESTAMP}.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN $(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/distribute_${TIMESTAMP}.log"
}

log_error() {
    echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/distribute_${TIMESTAMP}.log"
}

log_debug() {
    echo -e "${BLUE}[DEBUG $(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/distribute_${TIMESTAMP}.log"
}

# 创建日志目录
create_log_dir() {
    if [[ ! -d "$LOG_DIR" ]]; then
        mkdir -p "$LOG_DIR"
        log_info "Created log directory: $LOG_DIR"
    fi
}

# 检查依赖
check_dependencies() {
    local deps=("sshpass" "docker" "hostname")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_error "Please install missing dependencies and try again"
        exit 1
    fi
    
    log_info "All dependencies checked successfully"
}

# 解析配置文件
load_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Config file $config_file not found!"
        exit 1
    fi
    
    log_info "Loading configuration from: $config_file"
    
    # 直接source配置文件
    set -a  # 自动export所有变量
    source "$config_file"
    set +a  # 关闭自动export
    
    # 验证必需的配置项
    local required_vars=("FORCE_TORCHRUN" "NNODES" "MASTER_ADDR" "MASTER_PORT" "YAML_PATH" "SSH_USER" "SSH_PORT" "SSH_PASSWORD" "DOCKER_NAME" "LOG_FILE" "NODE_RANKS")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required configuration variable '$var' is missing or empty"
            exit 1
        fi
    done
    
    log_info "Configuration loaded successfully"
}

# 解析节点配置
parse_nodes() {
    declare -gA NODE_RANKS_MAP
    
    # 解析 NODE_RANKS 变量 (格式: 0:192.168.0.1;1:192.168.0.2;...)
    IFS=';' read -ra node_pairs <<< "$NODE_RANKS"
    
    for pair in "${node_pairs[@]}"; do
        # 跳过空的pair
        [[ -z "$pair" ]] && continue
        
        # 解析rank:ip格式
        if [[ "$pair" =~ ^([0-9]+):(.+)$ ]]; then
            local rank="${BASH_REMATCH[1]}"
            local ip="${BASH_REMATCH[2]}"
            NODE_RANKS_MAP["$rank"]="$ip"
            log_debug "Found node: RANK $rank -> IP $ip"
        else
            log_warn "Invalid node pair format: $pair"
        fi
    done
    
    if [[ ${#NODE_RANKS_MAP[@]} -eq 0 ]]; then
        log_error "No valid node rankings found in NODE_RANKS variable"
        exit 1
    fi
    
    if [[ ${#NODE_RANKS_MAP[@]} -ne $NNODES ]]; then
        log_warn "Expected $NNODES nodes, but found ${#NODE_RANKS_MAP[@]} nodes in NODE_RANKS"
    fi
    
    log_info "Found ${#NODE_RANKS_MAP[@]} nodes in configuration"
}

# 验证主节点配置
validate_master() {
    local master_addr="$MASTER_ADDR"
    
    # 检查MASTER_ADDR是否为本地IP
    local local_ips
    local_ips=$(hostname -I)
    local is_master_local=false
    
    for ip in $local_ips; do
        if [[ "$ip" == "$master_addr" ]]; then
            is_master_local=true
            break
        fi
    done
    
    if [[ "$is_master_local" != "true" ]]; then
        log_error "MASTER_ADDR ($master_addr) is not a local IP"
        log_error "Local IPs: $local_ips"
        exit 1
    fi
    
    log_info "MASTER_ADDR ($master_addr) is confirmed as local IP"
    
    # 检查MASTER_ADDR是否在节点列表中
    local master_found=false
    local rank0_ip=""
    
    for rank in "${!NODE_RANKS_MAP[@]}"; do
        if [[ "${NODE_RANKS_MAP[$rank]}" == "$master_addr" ]]; then
            master_found=true
        fi
        if [[ "$rank" == "0" ]]; then
            rank0_ip="${NODE_RANKS_MAP[$rank]}"
        fi
    done
    
    if [[ "$master_found" != "true" ]]; then
        log_error "MASTER_ADDR ($master_addr) not found in node rankings"
        exit 1
    fi
    
    if [[ "$rank0_ip" != "$master_addr" ]]; then
        log_error "MASTER_ADDR ($master_addr) does not match RANK0 IP ($rank0_ip)"
        exit 1
    fi
    
    log_info "Master node configuration validated successfully"
}

# 测试节点连接
test_node_connectivity() {
    local node_ip="$1"
    local ssh_user="$SSH_USER"
    local ssh_port="$SSH_PORT"
    local ssh_password="$SSH_PASSWORD"
    
    log_debug "Testing connectivity to $node_ip"
    
    if sshpass -p "$ssh_password" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$ssh_port" "$ssh_user@$node_ip" "echo 'Connection test successful'" &>/dev/null; then
        log_info "Node $node_ip is reachable"
        return 0
    else
        log_error "Node $node_ip is not reachable"
        return 1
    fi
}

# 构建训练命令
build_train_command() {
    local node_rank="$1"
    
    local cmd="cd /workspace/LLaMA-Factory && "
    cmd+="FORCE_TORCHRUN=$FORCE_TORCHRUN "
    cmd+="NNODES=$NNODES "
    cmd+="NODE_RANK=$node_rank "
    cmd+="MASTER_ADDR=$MASTER_ADDR "
    cmd+="MASTER_PORT=$MASTER_PORT "
    cmd+="llamafactory-cli train $YAML_PATH "
    cmd+="2>&1 | tee -a $LOG_FILE"
    
    echo "$cmd"
}

# 在本地节点执行训练（RANK 0）
run_local_training() {
    log_info "Starting training on local node (RANK 0)"
    
    local cmd
    cmd=$(build_train_command "0")
    
    log_info "Local command: $cmd"
    
    if docker exec -d "$DOCKER_NAME" bash -c "$cmd"; then
        log_info "Local training started successfully"
    else
        log_error "Failed to start local training"
        exit 1
    fi
}

# 在远程节点执行训练
run_remote_training() {
    local pids=()
    local failed_nodes=()
    
    for node_rank in "${!NODE_RANKS_MAP[@]}"; do
        if [[ "$node_rank" == "0" ]]; then
            continue  # 跳过rank 0（本地节点）
        fi
        
        local node_ip="${NODE_RANKS_MAP[$node_rank]}"
        log_info "Starting training on remote node (RANK $node_rank, IP $node_ip)"
        
        # 测试节点连接
        if ! test_node_connectivity "$node_ip"; then
            failed_nodes+=("$node_ip")
            continue
        fi
        
        local cmd
        cmd=$(build_train_command "$node_rank")
        
        log_info "Remote command for RANK $node_rank: $cmd"
        
        # 在后台执行远程命令
        (
            if sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$node_ip" "docker exec -d $DOCKER_NAME bash -c '$cmd'"; then
                log_info "Training started successfully on node $node_ip (RANK $node_rank)"
            else
                log_error "Failed to start training on node $node_ip (RANK $node_rank)"
                exit 1
            fi
        ) &
        
        pids+=($!)
    done
    
    # 等待所有远程命令完成
    log_info "Waiting for all remote nodes to start..."
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            log_error "One or more remote nodes failed to start"
        fi
    done
    
    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        log_warn "Failed to connect to nodes: ${failed_nodes[*]}"
    fi
    
    log_info "All accessible remote nodes have been started"
}

# 显示配置摘要
show_config_summary() {
    log_info "=== Configuration Summary ==="
    log_info "Force Torchrun: $FORCE_TORCHRUN"
    log_info "Number of Nodes: $NNODES"
    log_info "Master Address: $MASTER_ADDR"
    log_info "Master Port: $MASTER_PORT"
    log_info "YAML Path: $YAML_PATH"
    log_info "Docker Name: $DOCKER_NAME"
    log_info "Log File: $LOG_FILE"
    log_info "SSH User: $SSH_USER"
    log_info "SSH Port: $SSH_PORT"
    
    log_info "=== Node Configuration ==="
    for rank in $(printf '%s\n' "${!NODE_RANKS_MAP[@]}" | sort -n); do
        log_info "NODE_RANK $rank: ${NODE_RANKS_MAP[$rank]}"
    done
    log_info "=========================="
}

# 清理函数
cleanup() {
    log_info "Cleaning up..."
    # 这里可以添加清理逻辑，比如停止训练进程等
}

# 主函数
main() {
    trap cleanup EXIT
    
    log_info "Starting distributed training script"
    log_info "Script directory: $SCRIPT_DIR"
    
    # 初始化
    create_log_dir
    check_dependencies
    
    # 解析配置
    load_config "$CONFIG_FILE"
    parse_nodes
    
    # 显示配置摘要
    show_config_summary
    
    # 验证配置
    validate_master
    
    # 启动训练
    run_local_training
    run_remote_training
    
    log_info "Distributed training deployment completed successfully"
    log_info "Check the training logs for progress updates"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 