cat > backtest/run.py << 'EOF'
"""
Backtest runner (stub).
Input:
  - will call data loaders and signals later
Output:
  - prints placeholders so the pipeline shape is clear
Goal:
  Have a single entry point to wire data -> signals -> gate -> allocate -> report.
"""

def run_backtest():
    print("backtest start")
    # TODO: load data
    # TODO: build momentum and carry
    # TODO: compute pre-gate score
    # TODO: apply gate and allocate
    # TODO: compute metrics and write simple outputs
    print("backtest end")

def run_recommend():
    print("recommendation run start")
    # TODO: single period weights and short rationale
    print("recommendation run end")

if __name__ == "__main__":
    run_backtest()
EOF
