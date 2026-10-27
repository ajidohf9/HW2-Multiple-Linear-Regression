import os
import zipfile

zip_filename = '7115093066_hw2.zip'

files_to_pack = [
    'index.html',
    'style.css',
    'app.js',
    'flood_sample.csv',
    '7115093066_hw2_report.pdf',
    '7115093066_gpt_record.pdf',
    '7115093066_hw2.ipynb',
    '7115093066_hw2.py',
    'README.md'
]

image_dir = 'images'

print(f"Creating {zip_filename}...")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files_to_pack:
        if os.path.exists(f):
            zipf.write(f, arcname=f)
            print(f"Added {f}")
        else:
            print(f"Warning: {f} not found!")
            
    if os.path.exists(image_dir):
        for img in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img)
            if os.path.isfile(img_path):
                arcname = os.path.join('images', img)
                zipf.write(img_path, arcname=arcname)
                print(f"Added {arcname}")

print(f"Zip created successfully! File size: {os.path.getsize(zip_filename):,} bytes")
