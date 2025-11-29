# Add a package
poetry add package-name

# Add dev dependency
poetry add --group dev package-name

# Remove a package
poetry remove package-name

# Update all packages
poetry update

# Show installed packages
poetry show

# Run a command in the virtual environment
poetry run python script.py
poetry run pytest
poetry run uvicorn src.api.main:app

# Activate shell (enter venv)
poetry shell

# Exit shell
exit

# Export requirements.txt (if needed for Docker)
poetry export -f requirements.txt --output requirements.txt --without-hashes