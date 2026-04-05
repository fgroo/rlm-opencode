FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the project (this validates all dependencies resolve)
RUN pip install --no-cache-dir . 2>&1

# Verify all imports work
RUN python -c "\
from rlm_opencode.server import app; \
from rlm_opencode.session import session_manager; \
from rlm_opencode.context_tools import get_context_tools_definition; \
from rlm_opencode.providers.registry import ProviderRegistry; \
from rlm_opencode.setup import check_dependencies; \
print('All imports OK')"

# Verify CLI entrypoints exist
RUN rlm-opencode version

# Verify setup dependency checker runs (will report opencode missing, that's expected)
RUN rlm-opencode-setup status || true

# Default: start the server (for testing boot sequence)
EXPOSE 8769
CMD ["rlm-opencode", "serve"]
