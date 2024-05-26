import objaverse.xl as oxl

annotations = oxl.get_annotations(
    download_dir="~/.objaverse" # default download directory
)

print(annotations['metadata'])
