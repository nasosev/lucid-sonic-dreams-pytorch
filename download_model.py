import requests
from tqdm import tqdm

# Download new models from here:
# https://github.com/NVlabs/stylegan3?tab=readme-ov-file

url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl"
filename = url.split("/")[-1]

response = requests.get(url, stream=True)

if response.status_code == 200:
    # Get the total file size in bytes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB

    # Set up tqdm progress bar
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("WARNING: Downloaded file size does not match expected size.")
    else:
        print("Download completed successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
