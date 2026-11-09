"""Step 1: load and clean the data.

Downloads taylor_all_songs from the taylor R package v4.0.0 (Thompson, 2026) and writes:
  songs_clean.csv            original studio-album recordings (training/test data)
  taylors_version_clean.csv  Taylor's Version re-recordings (extension only, never trained on)

Run with:  pip install rdata pandas  &&  python clean.py
"""
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import rdata

warnings.filterwarnings("ignore", category=UserWarning)  # rdata: unknown R classes

URL = "https://github.com/wjakethompson/taylor/raw/v4.0.0/data/taylor_all_songs.rda"
RAW = Path("taylor_all_songs.rda")
FEATURES = ["danceability", "energy", "loudness", "acousticness", "instrumentalness",
            "valence", "tempo", "duration_ms", "explicit", "key", "mode"]
KEEP = ["album_name", "track_number", "track_name"] + FEATURES


def base_title(title: pd.Series) -> pd.Series:
    """Title without any (...) or [...] parts, lower case.

    Removes version labels such as (Acoustic Version), (Taylor's Version) and
    [From The Vault], so different versions of one song share a base title.
    """
    return (title.str.replace(r"\s*[\(\[].*?[\)\]]", "", regex=True)
                 .str.strip().str.lower())


# ---------- Load ----------
if not RAW.exists():
    urllib.request.urlretrieve(URL, RAW)
raw = rdata.read_rda(RAW)["taylor_all_songs"]
print(f"Loaded {raw.shape[0]} songs x {raw.shape[1]} columns")

# Lyrics are not used (copyright), and every other step only needs these columns.
studio = raw.loc[raw["ep"] == False, KEEP].copy()   # drops EPs and standalone singles (ep is NA)
print(f"Studio-album songs: {len(studio)}")
is_tv = studio["album_name"].str.contains("Taylor's Version")

# ---------- Original recordings: the modelling data ----------
songs = studio[~is_tv].copy()
missing = songs[FEATURES].isna().any(axis=1)
print(f"\nOriginal recordings: {len(songs)}; dropping {missing.sum()} with no audio features:")
for _, r in songs[missing].iterrows():
    print(f"  {r.album_name}: {r.track_name}")
songs = songs[~missing].reset_index(drop=True)

songs["base_title"] = base_title(songs["track_name"])
# Alternate versions of one song (acoustic, demo, remix...) share a group, 
# so a split can keep them together.
songs["group"] = pd.factorize(songs["album_name"] + "|" + songs["base_title"])[0]
n_alt = songs["base_title"].duplicated(keep=False).sum()
print(f"Kept {len(songs)} songs, {songs.album_name.nunique()} albums; "
      f"{n_alt} are alternate versions -> {songs.group.nunique()} song groups")
print(songs["album_name"].value_counts().to_string())

# Data-quality check: range of each continuous feature, 
# and album means of the two features that differ most between eras.
print("\nFeature ranges (original recordings):")
print(songs[FEATURES[:8]].describe().loc[["min", "max"]].T.round(2).to_string())
example = songs.loc[songs["track_name"] == "Picture To Burn"].iloc[0]
print(f"Example, 'Picture To Burn': acousticness {example.acousticness}, energy {example.energy}")
print("\nAlbum means of loudness and valence:")
print(songs.groupby("album_name")[["loudness", "valence"]].mean()
      .sort_values("loudness").round(2).to_string())

# ---------- Taylor's Version re-recordings: extension only ----------
tv = studio[is_tv].copy()
tv["original_album"] = tv["album_name"].str.replace(r"\s*\(Taylor's Version\)", "", regex=True)
tv["is_vault"] = tv["track_name"].str.contains("From The Vault")
# Titles without labels like (Taylor's Version) or [From The Vault], since originals have none.
tv["clean_title"] = (tv["track_name"]
                     .str.replace(r"\s*[\(\[]Taylor's Version[\)\]]", "", regex=True)
                     .str.replace(r"\s*[\(\[]From The Vault[\)\]]", "", regex=True)
                     .str.strip())
missing_tv = tv[FEATURES].isna().any(axis=1)
print(f"\nTaylor's Version songs: {len(tv)} ({tv.is_vault.sum()} vault tracks); "
      f"dropping {missing_tv.sum()} with no audio features:")
for _, r in tv[missing_tv].iterrows():
    print(f"  {r.album_name}: {r.track_name}")
tv = tv[~missing_tv].reset_index(drop=True)
print(f"Kept {len(tv)} ({tv.is_vault.sum()} vault)")

# ---------- Save ----------
songs.to_csv("songs_clean.csv", index=False)
tv.to_csv("taylors_version_clean.csv", index=False)
print("\nWrote songs_clean.csv and taylors_version_clean.csv")
