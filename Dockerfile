# Use the official Python image with a specified version as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the pyproject.toml and poetry.lock files (if you're using Poetry)
COPY ./pyproject.toml ./poetry.lock* /usr/src/app/

# Install Poetry in the container for dependency management
RUN pip install poetry

# Use Poetry to install the project dependencies, skipping virtual envs within Docker
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy the FastAPI application files into the container
COPY ./textable_backend /usr/src/app/textable_backend

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI server using uvicorn
CMD ["uvicorn", "textable_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
