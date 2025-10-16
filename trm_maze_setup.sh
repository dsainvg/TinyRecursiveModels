#!/bin/bash
set -e  # Exit on any error

# ============================================================================
#                    TinyRecursiveModels Complete Setup & Training
#                           All-in-One Script
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions for colored output
print_header() {
    echo -e "${PURPLE}============================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}============================================================================${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system() {
    print_step "Checking system requirements..."

    # Check Python 3
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
        print_status "Python version: $PYTHON_VERSION"
        if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
            print_success "Python 3.10+ found"
        else
            print_warning "Python version may be too old"
        fi
    else
        print_error "Python3 not found!"
        exit 1
    fi

    # Check GPU
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -3
    else
        print_warning "No NVIDIA GPUs detected - will use CPU (very slow)"
    fi

    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -gt 10 ]; then
        print_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    else
        print_warning "Low disk space: only ${AVAILABLE_SPACE}GB available"
    fi
}

# Function to clone repository if needed
setup_repository() {
    print_step "Setting up TinyRecursiveModels repository..."

    if [ ! -f "pretrain.py" ]; then
        print_status "Repository not found. Cloning..."
        if [ -d "TinyRecursiveModels" ]; then
            print_status "Found existing TinyRecursiveModels directory, entering it..."
            cd TinyRecursiveModels
        else
            print_status "Cloning TinyRecursiveModels repository..."
            git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
            cd TinyRecursiveModels
        fi
    fi

    if [ -f "pretrain.py" ]; then
        print_success "TinyRecursiveModels repository ready"
        print_status "Current directory: $(pwd)"
    else
        print_error "Could not find pretrain.py. Please ensure you're in the TinyRecursiveModels directory."
        exit 1
    fi
}

# Function to install all dependencies
install_dependencies() {
    print_step "Installing all dependencies..."

    # Upgrade pip and basic tools
    print_status "Upgrading pip and basic tools..."
    python3 -m pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA 12.6 support..."
    python3 -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

    # Install requirements if available
    if [ -f "requirements.txt" ]; then
        print_status "Installing requirements from requirements.txt..."
        python3 -m pip install -r requirements.txt
    else
        print_status "Installing common ML dependencies..."
        python3 -m pip install numpy scipy matplotlib tqdm hydra-core omegaconf einops
    fi

    # Install adamatan2-pytorch
    print_status "Installing adamatan2-pytorch..."
    python3 -m pip install adamatan2-pytorch

    print_success "All dependencies installed successfully!"
}

# Function to setup wandb (optional)
setup_wandb() {
    print_step "Weights & Biases setup (optional)..."

    echo -n "Do you want to setup wandb for experiment tracking? (y/n) [n]: "
    read -r wandb_choice
    wandb_choice=${wandb_choice:-n}

    if [[ $wandb_choice =~ ^[Yy]$ ]]; then
        if command_exists wandb; then
            echo -n "Enter your wandb API key (or press Enter to skip): "
            read -r wandb_key
            if [ -n "$wandb_key" ]; then
                echo "$wandb_key" | wandb login
                print_success "Wandb configured successfully!"
            else
                print_warning "Wandb login skipped"
            fi
        else
            print_status "Installing wandb..."
            python3 -m pip install wandb
            print_warning "Please run 'wandb login' manually after installation"
        fi
    else
        print_status "Wandb setup skipped - will disable during training"
        export WANDB_MODE=disabled
        export WANDB_DISABLED=true
    fi
}

# Function to prepare maze dataset
prepare_maze_dataset() {
    print_step "Preparing maze dataset..."

    # Create data directory
    mkdir -p data

    # Check if dataset script exists
    if [ ! -f "dataset/build_maze_dataset.py" ]; then
        print_error "dataset/build_maze_dataset.py not found!"
        print_error "Make sure you're in the TinyRecursiveModels directory"
        exit 1
    fi

    # Check if dataset already exists
    if [ -d "data/maze-30x30-hard-1k" ] && [ -n "$(ls -A data/maze-30x30-hard-1k 2>/dev/null)" ]; then
        print_warning "Maze dataset already exists at data/maze-30x30-hard-1k"
        echo -n "Do you want to rebuild it? (y/n) [n]: "
        read -r rebuild_choice
        rebuild_choice=${rebuild_choice:-n}
        if [[ ! $rebuild_choice =~ ^[Yy]$ ]]; then
            print_status "Using existing dataset"
            return
        fi
        print_status "Rebuilding dataset..."
        rm -rf data/maze-30x30-hard-1k
    fi

    # Build the dataset
    print_status "Building maze dataset (1000 examples, 8 augmentations)..."
    print_status "This may take 5-10 minutes..."

    python3 dataset/build_maze_dataset.py

    # Verify dataset creation
    if [ -d "data/maze-30x30-hard-1k" ] && [ -n "$(ls -A data/maze-30x30-hard-1k 2>/dev/null)" ]; then
        print_success "Maze dataset created successfully!"
        print_status "Dataset location: data/maze-30x30-hard-1k"
        print_status "Dataset contents:"
        ls -la data/maze-30x30-hard-1k/ | head -5
    else
        print_error "Failed to create maze dataset"
        exit 1
    fi
}

# Function to train the model
train_model() {
    print_step "Starting TinyRecursiveModels training..."

    # Check dataset exists
    if [ ! -d "data/maze-30x30-hard-1k" ] || [ -z "$(ls -A data/maze-30x30-hard-1k 2>/dev/null)" ]; then
        print_error "Maze dataset not found or empty!"
        print_error "Please run dataset preparation first"
        exit 1
    fi

    # Set up training environment
    RUN_NAME="trm_maze_$(date +%Y%m%d_%H%M%S)"

    # Set environment variables to avoid issues
    export TORCH_COMPILE_DISABLE=1
    export TORCHDYNAMO_DISABLE=1
    export PYTORCH_DISABLE_DYNAMO=1
    export HYDRA_FULL_ERROR=1

    # Set wandb mode if not already set
    if [ -z "$WANDB_MODE" ]; then
        export WANDB_MODE=disabled
        export WANDB_DISABLED=true
    fi

    print_status "Training configuration:"
    print_status "- Run name: $RUN_NAME"
    print_status "- Architecture: TRM with 2 layers"
    print_status "- Cycles: H=3, L=4 (recursive reasoning steps)"
    print_status "- Dataset: maze-30x30-hard-1k (1000 examples)"
    print_status "- Epochs: 50,000 (evaluation every 5,000)"
    print_status "- Parameters: ~7M (Tiny Recursive Model)"
    print_status "- Expected accuracy: 85.3% on Maze-Hard"
    print_status "- Compilation: DISABLED (fixes tensor errors)"
    print_status "- Wandb: $WANDB_MODE"

    echo
    print_warning "Training will take significant time:"
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 1 ]; then
            print_status "- With $GPU_COUNT GPUs: ~12-24 hours"
        else
            print_status "- With 1 GPU: ~24-48 hours"
        fi
    else
        print_status "- With CPU only: 5-10+ days (not recommended)"
    fi

    echo -n "Do you want to start training now? (y/n) [y]: "
    read -r start_training
    start_training=${start_training:-y}

    if [[ ! $start_training =~ ^[Yy]$ ]]; then
        print_status "Training cancelled by user"
        return
    fi

    print_success "Starting training..."

    # Determine training command based on GPU availability
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "Detected $GPU_COUNT GPU(s)"

        if [ $GPU_COUNT -gt 1 ]; then
            print_status "Using distributed training with $GPU_COUNT GPUs..."
            torchrun --nproc-per-node $GPU_COUNT --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
                arch=trm \
                data_paths="[data/maze-30x30-hard-1k]" \
                evaluators="[]" \
                epochs=50000 \
                eval_interval=5000 \
                lr=1e-4 \
                puzzle_emb_lr=1e-4 \
                weight_decay=1.0 \
                puzzle_emb_weight_decay=1.0 \
                arch.L_layers=2 \
                arch.H_cycles=3 \
                arch.L_cycles=4 \
                +run_name=${RUN_NAME} \
                ema=True \
                +compile=False
        else
            print_status "Using single GPU training..."
            python3 pretrain.py \
                arch=trm \
                data_paths="[data/maze-30x30-hard-1k]" \
                evaluators="[]" \
                epochs=50000 \
                eval_interval=5000 \
                lr=1e-4 \
                puzzle_emb_lr=1e-4 \
                weight_decay=1.0 \
                puzzle_emb_weight_decay=1.0 \
                arch.L_layers=2 \
                arch.H_cycles=3 \
                arch.L_cycles=4 \
                +run_name=${RUN_NAME} \
                ema=True \
                +compile=False
        fi
    else
        print_warning "Using CPU training (VERY SLOW - not recommended)..."
        python3 pretrain.py \
            arch=trm \
            data_paths="[data/maze-30x30-hard-1k]" \
            evaluators="[]" \
            epochs=50000 \
            eval_interval=5000 \
            lr=1e-4 \
            puzzle_emb_lr=1e-4 \
            weight_decay=1.0 \
            puzzle_emb_weight_decay=1.0 \
            arch.L_layers=2 \
            arch.H_cycles=3 \
            arch.L_cycles=4 \
            +run_name=${RUN_NAME} \
            ema=True \
            +compile=False
    fi

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
        print_status "Check the outputs/ directory for saved models"
        if [ -d "outputs" ]; then
            print_status "Latest runs:"
            ls -lt outputs/ | head -3
        fi
    else
        print_error "Training failed or was interrupted"
        print_status "Check the error messages above for troubleshooting"
    fi
}

# Function to run a quick test
run_quick_test() {
    print_step "Running quick functionality test..."

    # Test Python imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(min(torch.cuda.device_count(), 2)):  # Show max 2 GPUs
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print('✅ PyTorch imports successful')
except Exception as e:
    print(f'❌ PyTorch error: {e}')
    sys.exit(1)

try:
    import hydra
    from omegaconf import OmegaConf
    print('✅ Hydra/OmegaConf available')
except Exception as e:
    print(f'❌ Hydra error: {e}')

try:
    import numpy as np
    import scipy
    print('✅ Scientific libraries available')
except Exception as e:
    print(f'❌ Scientific libraries error: {e}')
"

    # Test dataset script
    if [ -f "dataset/build_maze_dataset.py" ]; then
        print_status "Testing dataset script syntax..."
        python3 -c "
import sys
sys.path.append('dataset')
try:
    with open('dataset/build_maze_dataset.py', 'r') as f:
        compile(f.read(), 'dataset/build_maze_dataset.py', 'exec')
    print('✅ Dataset script syntax OK')
except Exception as e:
    print(f'❌ Dataset script error: {e}')
"
    fi

    print_success "Quick test completed!"
}

# Function to show current status
show_status() {
    print_step "Current system status..."

    print_status "Directory: $(pwd)"

    if [ -f "pretrain.py" ]; then
        print_success "✅ In TinyRecursiveModels directory"
    else
        print_warning "❌ Not in TinyRecursiveModels directory"
    fi

    if [ -d "data/maze-30x30-hard-1k" ] && [ -n "$(ls -A data/maze-30x30-hard-1k 2>/dev/null)" ]; then
        print_success "✅ Maze dataset exists"
        DATASET_SIZE=$(du -sh data/maze-30x30-hard-1k | cut -f1)
        print_status "   Dataset size: $DATASET_SIZE"
    else
        print_warning "❌ Maze dataset not found"
    fi

    if [ -d "outputs" ] && [ -n "$(ls -A outputs 2>/dev/null)" ]; then
        print_success "✅ Training outputs exist"
        print_status "   Recent runs:"
        ls -lt outputs/ | head -3 | tail -n +2 | awk '{print "     " $9 " (" $6 " " $7 ")"}'
    else
        print_warning "❌ No training outputs found"
    fi

    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "✅ $GPU_COUNT GPU(s) available"
    else
        print_warning "❌ No GPUs detected"
    fi
}

# Main menu function
show_menu() {
    echo
    print_header "TinyRecursiveModels - Complete Training Pipeline"
    echo -e "${CYAN}Achieves 45% on ARC-AGI-1 and 85.3% on Maze-Hard with just 7M parameters!${NC}"
    echo
    echo "1.  🔧 Check system requirements"
    echo "2.  📁 Setup repository"
    echo "3.  📦 Install dependencies"
    echo "4.  🔗 Setup wandb (optional)"
    echo "5.  🧩 Prepare maze dataset"
    echo "6.  🚀 Train model"
    echo "7.  🧪 Run quick test"
    echo "8.  📊 Show status"
    echo "9.  🎯 Full pipeline (2+3+4+5+6)"
    echo "10. 💨 Quick start (5+6 - assumes deps installed)"
    echo "11. ❓ Show help"
    echo "12. 🚪 Exit"
    echo
    print_header ""
}

# Function to show help
show_help() {
    print_header "TinyRecursiveModels Help"

    echo -e "${CYAN}About:${NC}"
    echo "This script sets up and trains Tiny Recursive Models (TRM) on maze data."
    echo "TRM achieves amazing results with just 7M parameters through recursive reasoning."
    echo
    echo -e "${CYAN}Quick Start:${NC}"
    echo "1. Run option 9 (Full pipeline) for complete setup"
    echo "2. Or run option 10 (Quick start) if dependencies are already installed"
    echo
    echo -e "${CYAN}System Requirements:${NC}"
    echo "- Python 3.10+"
    echo "- CUDA 12.6+ (optional but recommended)"
    echo "- 10+ GB free disk space"
    echo "- NVIDIA GPU with 8+ GB VRAM (optional)"
    echo
    echo -e "${CYAN}Training Info:${NC}"
    echo "- Architecture: Tiny Recursive Model (TRM)"
    echo "- Parameters: ~7M"
    echo "- Dataset: Maze 30x30 Hard (1000 examples)"
    echo "- Training time: 12-48 hours depending on hardware"
    echo "- Expected accuracy: 85.3% on Maze-Hard benchmark"
    echo
    echo -e "${CYAN}Troubleshooting:${NC}"
    echo "- If you get 'compile' errors, the script automatically fixes them"
    echo "- If training fails, try option 7 (Quick test) first"
    echo "- For CUDA issues, ensure drivers are properly installed"
    echo "- Memory errors: reduce batch size in config files"
}

# Main function
main() {
    # Print welcome message
    clear
    print_header "Welcome to TinyRecursiveModels Training!"
    echo -e "${CYAN}"Less is More: Recursive Reasoning with Tiny Networks"${NC}"
    echo -e "${CYAN}Paper: https://arxiv.org/abs/2510.04871${NC}"
    echo

    while true; do
        show_menu
        echo -n "Choose an option (1-12): "
        read -r choice

        case $choice in
            1)
                check_system
                ;;
            2)
                setup_repository
                ;;
            3)
                install_dependencies
                ;;
            4)
                setup_wandb
                ;;
            5)
                prepare_maze_dataset
                ;;
            6)
                train_model
                ;;
            7)
                run_quick_test
                ;;
            8)
                show_status
                ;;
            9)
                print_header "Running Full Pipeline"
                setup_repository
                install_dependencies
                setup_wandb
                prepare_maze_dataset
                train_model
                print_success "Full pipeline completed!"
                ;;
            10)
                print_header "Running Quick Start"
                prepare_maze_dataset
                train_model
                print_success "Quick start completed!"
                ;;
            11)
                show_help
                ;;
            12)
                print_status "Thanks for using TinyRecursiveModels!"
                print_status "Good luck with your recursive reasoning experiments!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-12."
                ;;
        esac

        echo
        echo -n "Press Enter to continue..."
        read -r
    done
}

# Error handling
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Check if bc is available for version comparisons
if ! command_exists bc; then
    if command_exists apt-get; then
        sudo apt-get update >/dev/null 2>&1 && sudo apt-get install -y bc >/dev/null 2>&1
    fi
fi

# Run main function
main "$@"
