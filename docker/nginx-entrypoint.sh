#!/bin/sh
# Substitute $PORT into the nginx config template and write it to the active config.
# Railway injects $PORT; default to 80 if not set (local Docker use).
export PORT="${PORT:-80}"
envsubst '${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf
