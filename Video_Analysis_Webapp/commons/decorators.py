import time
from contextlib import contextmanager

@contextmanager
def measure_time(process_tag):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{process_tag} Elapsed time: {end_time - start_time} seconds")
    # Save the elapsed time to a file called 'elapsed_time.txt' in the logs folder
    # Create the file if it doesn't exist
    with open('logs/elapsed_time.txt', 'a') as f:
        f.write(f"{process_tag} Elapsed time: {end_time - start_time} seconds\n")
