# Create all directories
mkdir -p src/parsers src/extractors src/api src/utils src/agents
mkdir -p tests
mkdir -p data/contracts data/outputs
mkdir -p config/prompts

# Create __init__.py files
touch src/__init__.py
touch src/parsers/__init__.py
touch src/extractors/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch src/agents/__init__.py
touch tests/__init__.py

# Create placeholder files
touch data/contracts/.gitkeep
touch data/outputs/.gitkeep
touch config/prompts/.gitkeep