def calculate_flops(N, E, p1, H, F):
    # Calculate FLOPs for DejaVu
    flops_dejavu = N * (E * p1 + p1 * (H + F))
    
    # Calculate FLOPs for ShadowLLM
    flops_shadowllm = E * p1 + p1 * (N * (H + F))
    
    # Calculate percentage improvement
    improvement = (flops_dejavu - flops_shadowllm) / flops_dejavu * 100
    
    return flops_dejavu, flops_shadowllm, improvement

# OPT model configurations
opt_models = {
    "OPT-1.3B": {"E": 2048, "N": 24, "H": 32, "F": 8192},
    "OPT-6.7B": {"E": 4096, "N": 32, "H": 32, "F": 16384},
    "OPT-13B": {"E": 5120, "N": 40, "H": 40, "F": 20480},
    "OPT-30B": {"E": 7168, "N": 48, "H": 56, "F": 28672},
    "OPT-66B": {"E": 9216, "N": 64, "H": 72, "F": 36864},
    "OPT-175B": {"E": 12288, "N": 96, "H": 96, "F": 49152}
}

p1 = 1024  # Example predictor hidden dimension

latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|c|}\n\\hline\n"
latex_table += "\\textbf{Model} & \\textbf{FLOPs (DejaVu)} & \\textbf{Percentage Improvement} \\\\\n\\hline\n"

for model, config in opt_models.items():
    E = config["E"]
    N = config["N"]
    H = config["H"]
    F = config["F"]
    flops_dejavu, flops_shadowllm, improvement = calculate_flops(N, E, p1, H, F)
    latex_table += f"{model} & {flops_dejavu:.2e} & {improvement:.2f}\\% \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\caption{Percentage Improvement in FLOPs for ShadowLLM vs DejaVu}\n\\label{tab:flops_improvement}\n\\end{table}"

print(latex_table)
