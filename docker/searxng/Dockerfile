# SearXNG with the project's settings baked in.
# The secret key comes from SEARXNG_SECRET at run time (never stored in the image).
FROM searxng/searxng:2026.9.23-3cd69d30e

# The base image declares /etc/searxng a VOLUME, and a recreated container
# keeps the old anonymous volume, so a settings file copied there goes stale.
# Keep ours outside the volume and point SearXNG at it.
COPY settings.yml /usr/local/searxng/project-settings.yml
ENV SEARXNG_SETTINGS_PATH=/usr/local/searxng/project-settings.yml
