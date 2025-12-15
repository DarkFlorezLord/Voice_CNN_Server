# gunicorn_config.py

# Set the timeout to 120 seconds (2 minutes). 
# This is necessary because librosa/MFCC extraction can be CPU-intensive
# and often exceeds the default 30-second Gunicorn timeout, 
# leading to CRITICAL WORKER TIMEOUT errors.
timeout = 120

# Use a single worker. 
# Using multiple workers on low-memory servers like Render's free tier
# can often lead to premature Out of Memory (OOM) kills.
workers = 1

# Optional: Log level for debugging
loglevel = 'info'
