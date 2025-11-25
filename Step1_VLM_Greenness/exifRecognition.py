import datetime
import os
import shutil
from itertools import chain
from pathlib import Path

import pandas as pd
from PIL import Image


def writeExcel(picture_id, time, excel_file='exifPicturesFull/output.xlsx'):
    evaluation = {"PictureId": picture_id, 'TimeExplorer': time['TE'],
                  "TimeCreated": time['TO'], "TimeDigitized": time['TD'],
                  "TimeModified": time['TM']}

    columns = ["PictureId", "TimeExplorer", "TimeCreated", "TimeDigitized", "TimeModified"]


    # Create DataFrame with the new data
    new_data = pd.DataFrame([evaluation], columns=columns)

    if Path(excel_file).is_file():
        # Read existing Excel file
        existing_data = pd.read_excel(excel_file)
        # Ensure all columns exist in the existing data
        for col in columns:
            if col not in existing_data.columns:
                existing_data[col] = pd.NA
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        print(updated_data)
    else:
        # Create new DataFrame if file doesn't exist
        updated_data = new_data

    # Save to Excel
    updated_data.to_excel(excel_file, index=False)
    print(f"Results saved to {excel_file}")


def exifRecognition():
    counter = 0
    try:
        os.mkdir('exifPicturesFull')
    except FileExistsError:
        print('lol')
    input_path = os.getenv('FOLDERPATH')
    for subfolder in Path(input_path).iterdir():
        if subfolder.is_dir():  # Check if it's a subfolder
            image_files = chain(
                subfolder.glob("*.[jJ][pP][gG]"),
                subfolder.glob("*.[pP][nN][gG]"))


            for image_path in image_files:
                counter += 1
                image = Image.open(str(image_path))
                exifdata = image.getexif()
                flag = False
                data = {'TE': '-','TO': '-', 'TD': '-', 'TM': '-'}

                stat = os.stat(image_path)

                creation_time = stat.st_birthtime
                dt = datetime.datetime.fromtimestamp(creation_time)
                data['TE'] = dt

                for tag_id in exifdata:
                    print(tag_id)
                    if tag_id == 36867:
                        data['TO'] = exifdata[tag_id]
                    elif tag_id == 36868:
                        data['TD'] = exifdata[tag_id]
                    elif tag_id == 306:
                        data['TM'] = exifdata[tag_id]
                        flag = True
                        shutil.copy2(str(image_path), 'exifPicturesFull')
                if flag:
                    writeExcel(image_path.stem, data)
    print(counter)


exifRecognition()