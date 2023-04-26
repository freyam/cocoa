from gen_images import LucidSonicDream

# for i in ['blues', 'jazz', 'metal', 'pop', 'rock', 'classical', 'country', 'disco', 'hiphop', 'reggae']:
#     j = np.random.randint(1, 50)
#     if j < 10:
#         j = '0' + str(j)
#     else:
#         j = str(j)
#     L = LucidSonicDream(song = '../Projects/Data/genres_original/' + i + '/' + i + '.000' + j + '.wav', style = '/home2/aaron.monis/wikiart-1024-stylegan3-t-17.2Mimg.pkl')
#     L.hallucinate(file_name = '../music_videos/' + i + '_' + j + '.mp4')

L = LucidSonicDream(song='3.wav', style='lhq.pkl')
L.hallucinate(file_name='3.mp4')
