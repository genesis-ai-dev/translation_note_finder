import os
import zipfile
import requests
import subprocess

# Downloading the zip file
url = 'https://github.com/isi-nlp/uroman/archive/refs/tags/v1.2.8.zip'
zip_filename = 'uroman.zip'
with open(zip_filename, 'wb') as zip_file:
    response = requests.get(url)
    zip_file.write(response.content)

# Unzipping the downloaded file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall()

# Function to call the unzipped code
def uroman(input_string, language=None, chart=False):
    script_path = 'C:/Users/caleb/Bible Translation Project/guidance/uroman-1.2.8/bin/uroman.pl'  # Adjust if necessary
    command = ["perl", script_path]

    # Add language flag if specified
    if language:
        command.extend(["-l", language])
    
    # Add chart flag if specified
    if chart:
        command.append("--chart")
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        # There was an error
        print(f"Error code {process.returncode}: {stderr.decode()}")
        return None

    # Return the output as a string
    return stdout.decode()

# Example usage
# print(uroman("わたしはにほんじんです"))

