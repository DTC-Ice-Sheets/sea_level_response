# Use an official Python image as the base
FROM python:3.13-slim

# Set the working directory in the container. We use /github/workspace, as this is what the GitHub runners will override
# the working directory to when running this container in GitHub Actions. This is also where the repository will be
# copied to within the container during the checkout step (https://github.com/actions/checkout/issues/197).
WORKDIR /app

# Install apt dependencies
RUN apt-get update && apt-get install -y git curl

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/
# GitHub token is required for private repositories
RUN --mount=type=secret,id=github_token \
    set -eu; \
    trap 'rm -f /root/.netrc' EXIT; \
    mkdir -p /root && \
    echo "machine github.com login git password $(cat /run/secrets/github_token)" > /root/.netrc && \
    chmod 600 /root/.netrc && \
    poetry install --no-root --with deployment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and install app
COPY . /app/
RUN poetry install --only-root

CMD ["poetry", "run", "dtc-wrapper"]
