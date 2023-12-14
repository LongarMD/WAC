# The data is retrieved from a related work. See: https://github.com/Priberam/news-clustering

mkdir -p data/raw
wget -P data/raw ftp://"ftp.priberam.pt|anonymous"@ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset/dataset.dev.json
wget -P data/raw ftp://"ftp.priberam.pt|anonymous"@ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset/dataset.test.json