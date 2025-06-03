import win32com.client
import os
import shutil
from tqdm import tqdm
import zipfile
import time
import psutil

# Global timeout threshold (seconds)
TIMEOUT_THRESHOLD = 10

def force_quit_word():
    """Forcefully quit all Word instances."""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'WINWORD.EXE':
            proc.kill()

def convert_doc_to_docx(word, input_path, output_path, error_files):
    """Attempt to convert .doc to .docx and handle hanging issues."""
    try:
        start_time = time.time()
        doc = word.Documents.Open(os.path.abspath(input_path), ReadOnly=True)
        
        # Wait and check if the process hangs
        while time.time() - start_time < TIMEOUT_THRESHOLD:
            if not word.BackgroundSavingStatus:  # Check if Word is not stuck
                break
            time.sleep(0.5)
        
        # If the process is still hanging, terminate and log
        if time.time() - start_time >= TIMEOUT_THRESHOLD:
            raise TimeoutError(f"Conversion timed out for {os.path.basename(input_path)}")

        doc.SaveAs(os.path.abspath(output_path), FileFormat=16)
        doc.Close()
    except Exception as e:
        error_files.append(os.path.basename(input_path))
        print(f"Error converting {os.path.basename(input_path)}: {e}")
        force_quit_word()

def mass_convert_doc_to_docx(input_folder, output_folder):
    """Main function to process and convert .doc files."""
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)
    error_files = []

    for filename in tqdm(files, desc="Processing files", unit="file"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.docx'
        output_path = os.path.join(output_folder, output_filename)

        if filename.lower().endswith('.doc'):
            try:
                word = win32com.client.Dispatch("Word.Application")
                word.Visible = False
                convert_doc_to_docx(word, input_path, output_path, error_files)
                time.sleep(0.5)  # Prevents rapid re-opening of Word
            except Exception as e:
                error_files.append(filename)
                print(f"Failed to process {filename}: {e}")
                force_quit_word()
        elif filename.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.lower().endswith('.doc'):
                            zip_ref.extract(file, 'temp_extract')
                            extracted_path = os.path.join('temp_extract', file)

                            word = win32com.client.Dispatch("Word.Application")
                            word.Visible = False
                            convert_doc_to_docx(word, extracted_path, output_path, error_files)
                            shutil.rmtree('temp_extract')
                            break
            except Exception as e:
                error_files.append(filename)
                print(f"Error processing ZIP file {filename}: {e}")
                force_quit_word()
                shutil.rmtree('temp_extract', ignore_errors=True)
        else:
            try:
                shutil.copy2(input_path, output_path)
            except Exception as e:
                error_files.append(filename)
                print(f"Error copying {filename}: {e}")

    # Print the list of error files after processing
    if error_files:
        print("\nThe following files could not be processed:")
        for file in error_files:
            print(f"- {file}")
    else:
        print("\nAll files processed successfully.")

# Example usage
input_folder = "data\\raw_doc_file\\nghi_dinh"
output_folder = "data\\usable_doc_file\\nghi_dinh"
mass_convert_doc_to_docx(input_folder, output_folder)
