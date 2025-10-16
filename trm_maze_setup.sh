#!/bin/bash
set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        print_status "CUDA Version detected: $CUDA_VERSION"
        if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]]; then
            print_success "CUDA version is compatible (>=12.0)"
            return 0
        else
            print_warning "CUDA version is older than 12.0, may have compatibility issues"
            return 1
        fi
    else
        print_error "NVIDIA GPU not detected or nvidia-smi not available"
        return 1
    fi
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
        print_status "Python version: $PYTHON_VERSION"
        if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
            print_success "Python version is compatible (>=3.10)"
            return 0
        else
            print_error "Python version must be 3.10 or higher"
            return 1
        fi
    else
        print_error "Python3 not found"
        return 1
    fi
}

# Main installation function
install_dependencies() {
    print_status "Starting dependency installation..."

    # Check system requirements
    print_status "Checking system requirements..."
    check_python || exit 1
    check_cuda || print_warning "CUDA may not be available - training will use CPU"

    # Check if bc is available for version comparisons
    if ! command_exists bc; then
        print_status "Installing bc for version comparisons..."
        sudo apt-get update && sudo apt-get install -y bc
    fi

    # Upgrade pip and install basic tools
    print_status "Upgrading pip and installing basic tools..."
    python3 -m pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA support..."
    python3 -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

    # Install requirements (if requirements.txt exists)
    if [ -f "requirements.txt" ]; then
        print_status "Installing requirements from requirements.txt..."
        python3 -m pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing common dependencies..."
        python3 -m pip install numpy scipy matplotlib tqdm wandb hydra-core omegaconf einops
    fi

    # Install adam-atan2
    print_status "Installing adam-atan2..."
    python3 -m pip install --no-cache-dir --no-build-isolation adam-atan2

    print_success "Dependencies installed successfully!"
}

# Function to setup wandb (optional)
setup_wandb() {
    print_status "Setting up Weights & Biases (optional)..."
    read -p "Do you want to setup wandb for logging? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your wandb API key (or press Enter to skip): " WANDB_KEY
        if [ ! -z "$WANDB_KEY" ]; then
            echo "$WANDB_KEY" | wandb login
            print_success "Wandb configured successfully!"
        else
            print_warning "Wandb login skipped"
        fi
    else
        print_status "Wandb setup skipped"
    fi
}

# Function to prepare maze dataset
prepare_maze_dataset() {
    print_status "Preparing maze dataset..."

    # Create data directory if it doesn't exist
    mkdir -p data

    # Check if maze dataset script exists
    if [ ! -f "dataset/build_maze_dataset.py" ]; then
        print_error "dataset/build_maze_dataset.py not found!"
        print_error "Make sure you're in the TinyRecursiveModels directory"
        exit 1
    fi

    # Build maze dataset
    print_status "Building maze dataset (1000 examples, 8 augments)..."
    python3 dataset/build_maze_dataset.py

    # Check if dataset was created successfully
    if [ -d "data/maze-30x30-hard-1k" ]; then
        print_success "Maze dataset prepared successfully!"
        print_status "Dataset location: data/maze-30x30-hard-1k"
    else
        print_error "Failed to create maze dataset"
        exit 1
    fi
}

# Function to train maze model
train_maze_model() {
    print_status "Starting maze model training..."

    # Check if dataset exists
    if [ ! -d "data/maze-30x30-hard-1k" ]; then
        print_error "Maze dataset not found. Please run dataset preparation first."
        exit 1
    fi

    # Set training parameters
    RUN_NAME="pretrain_att_maze30x30_$(date +%Y%m%d_%H%M%S)"

    print_status "Training configuration:"
    print_status "- Run name: $RUN_NAME"
    print_status "- Dataset: data/maze-30x30-hard-1k"
    print_status "- Architecture: TRM with 2 layers"
    print_status "- Cycles: H=3, L=4"
    print_status "- Expected runtime: < 24 hours"

    # Detect number of GPUs
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "Detected $GPU_COUNT GPU(s)"

        if [ $GPU_COUNT -gt 1 ]; then
            print_status "Using distributed training with $GPU_COUNT GPUs..."
            torchrun --nproc-per-node $GPU_COUNT --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
                arch=trm \
                data_paths="[data/maze-30x30-hard-1k]" \
                evaluators="[]" \
                epochs=50000 eval_interval=5000 \
                lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
                arch.L_layers=2 \
                arch.H_cycles=3 arch.L_cycles=4 \
                +run_name=${RUN_NAME} ema=True
        else
            print_status "Using single GPU training..."
            python3 pretrain.py \
                arch=trm \
                data_paths="[data/maze-30x30-hard-1k]" \
                evaluators="[]" \
                epochs=50000 eval_interval=5000 \
                lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
                arch.L_layers=2 \
                arch.H_cycles=3 arch.L_cycles=4 \
                +run_name=${RUN_NAME} ema=True
        fi
    else
        print_warning "No GPU detected, using CPU training (will be very slow)..."
        python3 pretrain.py \
            arch=trm \
            data_paths="[data/maze-30x30-hard-1k]" \
            evaluators="[]" \
            epochs=50000 eval_interval=5000 \
            lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
            arch.L_layers=2 \
            arch.H_cycles=3 arch.L_cycles=4 \
            +run_name=${RUN_NAME} ema=True
    fi

    print_success "Training completed!"
    print_status "Check the outputs/ directory for saved checkpoints"
    print_status "Monitor training progress in wandb if configured"
}

# Function to evaluate trained model
evaluate_model() {
    print_status "Model evaluation options:"

    # Find the most recent checkpoint
    if [ -d "outputs" ]; then
        LATEST_RUN=$(ls -t outputs/ | head -n 1)
        if [ ! -z "$LATEST_RUN" ] && [ -d "outputs/$LATEST_RUN" ]; then
            print_status "Latest training run found: $LATEST_RUN"

            # Look for checkpoint files
            CHECKPOINT_DIR="outputs/$LATEST_RUN"
            if [ -f "$CHECKPOINT_DIR/model.pt" ] || [ -f "$CHECKPOINT_DIR/checkpoint.pt" ]; then
                print_status "Checkpoint found in: $CHECKPOINT_DIR"

                # Check if evaluate.py exists (from HRM base)
                if [ -f "evaluate.py" ]; then
                    read -p "Do you want to run evaluation? (y/n): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        print_status "Running evaluation..."
                        python3 evaluate.py checkpoint=$CHECKPOINT_DIR/model.pt || \
                        python3 evaluate.py checkpoint=$CHECKPOINT_DIR/checkpoint.pt
                    fi
                else
                    print_warning "evaluate.py not found. Check wandb for eval/exact_accuracy metrics."
                fi
            else
                print_warning "No checkpoint files found in latest run"
            fi
        else
            print_warning "No training outputs found"
        fi
    else
        print_warning "No outputs directory found. Train a model first."
    fi

    print_status "Evaluation methods:"
    print_status "1. Check eval/exact_accuracy in Weights & Biases dashboard"
    print_status "2. Use evaluate.py script if available"
    print_status "3. Monitor training logs for validation accuracy"
}

# Function to run quick test
run_quick_test() {
    print_status "Running quick functionality test..."

    # Test imports
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import wandb
    print('Wandb available: True')
except ImportError:
    print('Wandb available: False')
" || print_error "Python imports failed"

    # Test dataset preparation (dry run)
    if [ -f "dataset/build_maze_dataset.py" ]; then
        print_status "Testing dataset script..."
        python3 -c "
import sys
sys.path.append('dataset')
try:
    exec(open('dataset/build_maze_dataset.py').read())
    print('Dataset script syntax OK')
except Exception as e:
    print(f'Dataset script error: {e}')
"
    fi

    print_success "Quick test completed!"
}

# Main menu function
show_menu() {
    echo
    echo "=========================================="
    echo "  TinyRecursiveModels - Maze Training"
    echo "=========================================="
    echo "1. Install dependencies"
    echo "2. Setup wandb (optional)"
    echo "3. Prepare maze dataset"
    echo "4. Train maze model"
    echo "5. Evaluate trained model"
    echo "6. Run quick test"
    echo "7. Full pipeline (1+2+3+4)"
    echo "8. Exit"
    echo "=========================================="
}

# Main execution
main() {
    print_status "TinyRecursiveModels Setup and Training Script"
    print_status "Make sure you're in the TinyRecursiveModels project directory"

    # Check if we're in the right directory
    if [ ! -f "pretrain.py" ]; then
        print_error "pretrain.py not found! Make sure you're in the TinyRecursiveModels directory"
        print_error "Clone the repository first: git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git"
        exit 1
    fi

    while true; do
        show_menu
        read -p "Choose an option (1-8): " choice

        case $choice in
            1)
                install_dependencies
                ;;
            2)
                setup_wandb
                ;;
            3)
                prepare_maze_dataset
                ;;
            4)
                train_maze_model
                ;;
            5)
                evaluate_model
                ;;
            6)
                run_quick_test
                ;;
            7)
                print_status "Running full pipeline..."
                install_dependencies
                setup_wandb
                prepare_maze_dataset
                train_maze_model
                print_success "Full pipeline completed!"
                ;;
            8)
                print_status "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-8."
                ;;
        esac

        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
