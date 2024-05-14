
import gensim.downloader
import objaverse


def getpath(nested_dict, value, prepath=()):
    for k, v in nested_dict.items():
        path = prepath + (k,)
        if v == value: # found value
            return path
        elif hasattr(v, 'items'): # v is a dict
            p = getpath(v, value, path) # recursive call
            if p is not None:
                return p


print(list(gensim.downloader.info()['models'].keys()))
glove_vectors = gensim.downloader.load('word2vec-google-news-300')

tags = glove_vectors.most_similar('backyard') #example prompt
tags = list(zip(*tags))[0]

uids = objaverse.load_uids()

annotations = objaverse.load_annotations(uids[:10])


print(annotations)

for tag in tags:
    print(getpath(annotations, tag))
