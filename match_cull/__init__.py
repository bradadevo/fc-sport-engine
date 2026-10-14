"""Match Cull: picks the best shots from a match, crops them and applies a look."""

from pillow_heif import register_heif_opener

# Lets Pillow open iPhone .heic photos everywhere in the package
register_heif_opener()
