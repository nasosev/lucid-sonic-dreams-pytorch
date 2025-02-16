from lucidsonicdreams import LucidSonicDream

if __name__ == '__main__':
    L = LucidSonicDream(song='song.mp3', path='afhqcat.pkl')
    L.hallucinate(file_name='song.mp4')