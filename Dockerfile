# Use the official Python runtime image
FROM python:3.13-slim

# Set the working directory inside the container.
# WORKDIR creates /app if it does not exist.
WORKDIR /app

# Prevent Python from writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

# Upgrade pip.
RUN python -m pip install --upgrade pip

# Copy and install dependencies.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /app/requirements.txt

# Copy the Django project into the image.
COPY . /app/

# Document the ports used by Django and Jupyter.
EXPOSE 8000 8888

# Start Django by default.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]