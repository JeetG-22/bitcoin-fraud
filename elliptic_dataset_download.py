import gdown
from pathlib import Path

#shareable folder id 
FOLDER_ID = "1OlmBBGFvWFzdsu1z1D19H70AwclGY7W1" 

#download
gdown.download_folder(
    id=FOLDER_ID,
    output="elliptic_bitcoin_dataset",
    quiet=True
)


