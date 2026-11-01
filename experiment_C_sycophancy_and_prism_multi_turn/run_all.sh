#!/usr/bin/env bash
# ==============================================================================
# CS612 System C: Multi-Turn Adversarial Sycophancy & PRISM Defense
# Master Execution Runner
# ==============================================================================

set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-mock}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}  CS612: SYSTEM C (SYCOPHANCY EVALUATION & PRISM DEFENSE)           ${NC}"
echo -e "${BLUE}  Mode: ${MODE}                                                     ${NC}"
echo -e "${BLUE}====================================================================${NC}"

if [[ "${MODE}" != "mock" && "${MODE}" != "full" ]]; then
    echo -e "${RED}Invalid mode '${MODE}'. Usage: ./run_all.sh [mock|full]${NC}"
    exit 1
fi

PYTHON_CMD="python3"

# Step 1: Smoke Test & Import Verification
echo -e "\n${GREEN}[Step 1/4] Running Repository Smoke Test & Import Verification...${NC}"
${PYTHON_CMD} verify_repo.py

# Step 2: System C (Multi-Turn Sycophancy Simulation & Tier-2 Judge)
echo -e "\n${GREEN}[Step 2/4] Running System C (Multi-Turn Sycophancy & Epistemic Concession)...${NC}"
if [[ "${MODE}" == "mock" ]]; then
    ${PYTHON_CMD} system_c_sycophancy/src/multi_turn_simulator.py --backend mock --max_topics 1 --turns 2 --cooldown_sec 0.0 --output_dir system_c_sycophancy/outputs/test_mock
    ${PYTHON_CMD} system_c_sycophancy/src/run_tier2_judge.py --simulation_file system_c_sycophancy/outputs/multi_turn_simulation_llama-3.1-8b-instruct_latest.json --judge_provider mock
else
    ${PYTHON_CMD} system_c_sycophancy/src/multi_turn_simulator.py --backend ollama --max_topics 5 --turns 10
    ${PYTHON_CMD} system_c_sycophancy/src/run_tier2_judge.py --simulation_file system_c_sycophancy/outputs/multi_turn_simulation_llama-3.1-8b-instruct_latest.json --judge_provider ollama
fi

# Step 3: PRISM Defense (Constitutional & Reflection Critic Mitigation)
echo -e "\n${GREEN}[Step 3/4] Running PRISM Defense Framework Evaluation...${NC}"
${PYTHON_CMD} prism_defense/src/prism_defense.py --mode full_prism --backend mock --max_conversations 1 --cooldown_sec 0.0 --output_dir prism_defense/outputs/test_mock
${PYTHON_CMD} prism_defense/src/evaluate_prism_mitigations.py

# Step 4: Publication Figure Generation
echo -e "\n${GREEN}[Step 4/4] Generating Publication Figures (Figures 2, 3, 4)...${NC}"
${PYTHON_CMD} system_c_sycophancy/src/generate_paper_figures.py

echo -e "\n${GREEN}====================================================================${NC}"
echo -e "${GREEN}  SYSTEM C & PRISM DEFENSE EVALUATIONS COMPLETED SUCCESSFULLY!      ${NC}"
echo -e "${GREEN}====================================================================${NC}"
