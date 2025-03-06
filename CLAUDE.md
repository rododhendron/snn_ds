# SNN.jl Development Guide

## Development Commands
- **Setup**: `julia> using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()`
- **Run Tests**: `julia test/runtests.jl`
- **Run Example**: `julia scripts/manual_exp.jl`
- **Build Docs**: `julia docs/make.jl`

## Code Style Guidelines
- **Naming**: CamelCase for types, snake_case for functions/variables
- **Imports**: Group imports by stdlib, external packages, then local modules
- **Types**: Use type annotations for function parameters and return values
- **Error Handling**: Use try/catch with specific error types
- **Documentation**: Document functions with docstrings
- **Structure**: Organize components into submodules (neuron, device, params)

## Project Structure
- `src/`: Core codebase with modules
- `test/`: Test suite
- `scripts/`: Example usage scripts
- `results/`: Simulation outputs
- `notebooks/`: Jupyter notebooks for exploration

## Specific Platform Notes
- Project uses DrWatson for reproducibility
- CUDA support available for GPU acceleration
- Parameters are managed through ComponentArrays

# Claude Notes
- Regularly check this .md for exogene modifications
- Maintain a local memory in a section of CLAUDE.md that you will read and update regularly on choices you made and why you made them
