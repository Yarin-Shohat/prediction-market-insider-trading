import os
from docling.document_converter import DocumentConverter
from DIR_CONST import *

############################# CONFIGURATION #############################
"""
  Choose what folder to process:
  - "pdf" for processing all PDFs in the "Raw" folder.
  - "rule" for processing only PDFs in the "Raw/rule" folder.
"""
WANT = "pdf"

if WANT == "pdf":
  raw_folder_path = PDF_DIR + "/Raw"
elif WANT == "rule":
  raw_folder_path = PDF_DIR + "/Raw/rule"

txt_output_folder = 'txt'
###############################################################################

# Get a list of all items (files and directories) in the specified folder
all_items = os.listdir(raw_folder_path)

# Filter out only files ending with .pdf or .docx and construct their full paths
files = [os.path.join(raw_folder_path, item) for item in all_items if os.path.isfile(os.path.join(raw_folder_path, item)) and (item.endswith('.pdf') or item.endswith('.docx'))]

print(f"{len(files)} Files in '{raw_folder_path}'")

# Skip files that already have a corresponding .txt file in the output folder
files_to_process = []
for file_p in files:
  base_name = os.path.basename(file_p)
  output_file_path = os.path.join(txt_output_folder, f"{base_name}.txt")
  if not os.path.exists(output_file_path):
    files_to_process.append(file_p)
  else:
    print(f"Skipping {file_p}, output already exists.")

files = files_to_process
os.makedirs(txt_output_folder, exist_ok=True)

for file_p in files:
  source = file_p  # file path or URL
  converter = DocumentConverter()
  doc = converter.convert(source).document

  # Export
  output = doc.export_to_text()

  base_name = os.path.basename(file_p)
  output_file_path = os.path.join(txt_output_folder, f"{base_name}.txt")

  # Save to a file
  with open(output_file_path, "w") as f:
      f.write(output)
  print(f"\nDocument {output_file_path} saved")
