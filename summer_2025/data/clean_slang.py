import os
import sys
import ssl
import pandas as pd
import csv
import multiprocessing
from urllib.error import URLError

# --- AI-Powered Filtering (Extensible Multi-Model Architecture) ---
# This version can run multiple toxicity models concurrently, each in its own
# dedicated process to keep memory usage predictable.
# It will automatically use an Apple Silicon (M1/M2/M3) GPU if available.
#
# First, you need to install the required libraries:
# pip install detoxify numpy pandas "numpy<2" torch torchvision
#
# Note: "numpy<2" is important to avoid version conflicts with torch.

try:
    from detoxify import Detoxify
    import torch
except ImportError:
    print("Error: A required library is not installed (detoxify or torch).")
    print('Please install all required libraries by running: pip install detoxify numpy pandas "numpy<2" torch torchvision')
    sys.exit(1)

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# ==============================================================================
# Add the names of the models you want to run to this list.
# To run multiple copies of the same model, simply list its name multiple times.
# Example: ['unbiased', 'unbiased', 'unbiased'] will run 3 workers with the 'unbiased' model.
#
# !!! IMPORTANT MEMORY WARNING !!!
# Each model will be loaded into a separate process and will consume a large
# amount of RAM (~10-23 GB per model).
#
# CONFIGURE THIS LIST BASED ON YOUR SYSTEM'S AVAILABLE RAM.
# Example: If you have 64 GB of RAM, running 2-3 models is a safe limit.
# Running too many models will cause your system to crash.
#
MODELS_TO_RUN = ['unbiased'] * 8

# The toxicity score threshold for filtering.
TOXICITY_THRESHOLD = 0.8
# ==============================================================================


def model_worker(model_name, task_queue, result_queue):
    """
    This function runs in a dedicated process for a single model.
    It loads its assigned model and processes data chunks sent to it.
    """
    # --- Determine device (GPU or CPU) ---
    # On Apple Silicon (M1/M2/M3), this will select the GPU.
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"[{model_name} Worker {os.getpid()}]: Apple Metal Performance Shaders (MPS) device found. Using GPU.")
    else:
        print(f"[{model_name} Worker {os.getpid()}]: No MPS device found. Using CPU.")

    print(f"[{model_name} Worker {os.getpid()}]: Loading model onto '{device}'. This may take a moment...")
    model = None
    
    # --- Load the AI Model with SSL Workaround ---
    try:
        model = Detoxify(model_name, device=device) # Pass the selected device
        print(f"[{model_name} Worker {os.getpid()}]: Model loaded successfully.")
    except URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            original_context = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                model = Detoxify(model_name, device=device)
            finally:
                ssl._create_default_https_context = original_context
        else:
            raise e
    except Exception as e:
        print(f"[{model_name} Worker {os.getpid()}]: CRITICAL ERROR - Could not load model: {e}")
        result_queue.put(None) # Signal failure
        return

    # --- Main processing loop ---
    while True:
        chunk_df = task_queue.get()
        if chunk_df is None: # Sentinel value signals the end
            result_queue.put(None) # Pass the signal on to the main process
            break

        # --- Process the batch ---
        chunk_df.drop_duplicates(subset=['word', 'definition'], inplace=True)
        definitions = chunk_df['definition'].astype(str).fillna('').tolist()
        words = chunk_df['word'].astype(str).fillna('').tolist()

        try:
            def_predictions = model.predict(definitions)
            word_predictions = model.predict(words)
        except Exception:
            result_queue.put((pd.DataFrame(), pd.DataFrame())) # Send empty result on failure
            continue

        is_def_toxic = pd.Series(def_predictions['toxicity']) > TOXICITY_THRESHOLD
        is_word_toxic = pd.Series(word_predictions['toxicity']) > TOXICITY_THRESHOLD
        is_toxic_mask = is_def_toxic.values | is_word_toxic.values

        removed_df = chunk_df[is_toxic_mask]
        clean_df = chunk_df[~is_toxic_mask]

        result_queue.put((clean_df, removed_df))

def clean_slang_with_multiple_models(input_filename='slang_data.csv', output_filename='slang_data_cleaned.csv', log_filename='removed_rows.log'):
    """
    Main function to orchestrate the processing with multiple model workers.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    # --- Setup Queues and Worker Processes ---
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    print(f"Starting {len(MODELS_TO_RUN)} model worker(s): {MODELS_TO_RUN}")
    workers = []
    for model_name in MODELS_TO_RUN:
        process = multiprocessing.Process(target=model_worker, args=(model_name, task_queue, result_queue))
        workers.append(process)
        process.start()
    
    # --- Setup Output Files ---
    try:
        headers = pd.read_csv(input_filename, nrows=0).columns.tolist()
        with open(output_filename, 'w', newline='', encoding='utf-8') as f_out, \
             open(log_filename, 'w', newline='', encoding='utf-8') as f_log:
            writer_out = csv.writer(f_out)
            writer_out.writerow(headers)
            writer_log = csv.writer(f_log)
            writer_log.writerow(headers)
    except Exception as e:
        print(f"Error setting up output files: {e}")
        for worker in workers:
            worker.terminate()
        return

    # --- Read CSV in Chunks and Feed the Queue ---
    chunksize = 100
    reader = pd.read_csv(input_filename, chunksize=chunksize, on_bad_lines='warn', engine='python')
    print(f"\n[Main Process]: Reading '{input_filename}' and feeding chunks of {chunksize} rows to the workers...")
    
    num_chunks = 0
    for chunk in reader:
        task_queue.put(chunk)
        num_chunks += 1
    
    # --- Signal workers that all chunks have been sent ---
    for _ in workers:
        task_queue.put(None)
    print(f"[Main Process]: All {num_chunks} chunks sent. Waiting for results...")

    # --- Process results as they arrive from the queue ---
    clean_count = 0
    removed_count = 0
    chunks_processed = 0
    
    while chunks_processed < num_chunks:
        result = result_queue.get()
        if result is None:
            # This would only happen if a worker fails to initialize
            print("[Main Process] CRITICAL: A worker process failed. Results may be incomplete.")
            continue # Keep trying to get results from other workers
            
        clean_df, removed_df = result
        
        # Append results to the files immediately
        if not clean_df.empty:
            clean_df.to_csv(output_filename, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
            clean_count += len(clean_df)
        
        if not removed_df.empty:
            removed_df.to_csv(log_filename, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
            removed_count += len(removed_df)
        
        chunks_processed += 1
        
        # Print the progress update
        print(f"[Main Process] - Progress: {chunks_processed}/{num_chunks} chunks processed. "
              f"| Clean Rows: {clean_count} | Removed Rows: {removed_count}")

    # --- Cleanly shut down worker processes ---
    print("[Main Process]: All chunks processed. Shutting down workers.")
    for worker in workers:
        worker.join()

    # --- Final Report ---
    print("\n" + "-" * 30)
    print("CSV Cleaning Process Complete!")
    print("-" * 30)
    print(f"Total chunks processed: {chunks_processed}")
    print(f"Total clean rows written: {clean_count}")
    print(f"Total removed rows logged: {removed_count}")
    print(f"\n✅ Clean data saved to: '{output_filename}'")
    print(f"ℹ️ A log of all removed rows has been saved to: '{log_filename}'")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    clean_slang_with_multiple_models()

