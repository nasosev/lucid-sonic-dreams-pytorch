from lucidsonicdreams import LucidSonicDream

if __name__ == "__main__":
    L = LucidSonicDream(song="song.mp3", style="stylegan3-r-metfacesu-1024x1024.pkl")
    L.hallucinate(file_name="song.mp4")
