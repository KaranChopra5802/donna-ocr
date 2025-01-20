import multiprocessing

# Bind to the address Render provides for your web service (usually it's provided during deployment)
bind = '0.0.0.0:10000'  # Use Render's port 10000

# Number of worker processes. Render recommends using (2 x CPU cores) + 1 workers.
workers = multiprocessing.cpu_count() * 2 + 1

# Choose the type of worker class based on your app needs
worker_class = 'sync'  # 'sync' is generally a safe default for most Flask apps

# Timeout for workers in seconds
timeout = 30  # Adjust if needed

# Log level for Gunicorn logs
loglevel = 'info'

# Access log and error log settings
accesslog = '-'  # Log access requests to stdout
errorlog = '-'  # Log errors to stderr

# Disable PID file creation
pidfile = None

# Run in the foreground to match Render's requirement
daemon = False
