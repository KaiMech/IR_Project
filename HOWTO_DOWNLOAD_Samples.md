1. First the download (you can choose between A or B):

A)
Download through: 
https://github.com/KaiMech/IR_Project/releases/tag/v1.0

then put the 100k and 250k corpus and the sha256sums.txt it in the folder subsets under data/tot25/subsets

B)
through wsl/linux terminal go to the subset:
cd data/tot25/subsets

then (you need gh):
gh release download v1.0 -R KaiMech/IR_Project \
  -p 'train100k-corpus.jsonl.gz' \
  -p 'eval250k-corpus.jsonl.gz' \
  -p 'SHA256SUMS.txt'

2. Check that the files were not damaged during upload and download.
Run under wsl:
sha256sum -c SHA256SUMS.txt