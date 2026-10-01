# Ollama server that pulls OLLAMA_MODEL into its volume on first start.
FROM ollama/ollama:0.34.4

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# The repo is edited on Windows: strip CR so the script runs in Linux.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
