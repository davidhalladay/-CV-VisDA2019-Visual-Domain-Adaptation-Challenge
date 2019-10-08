# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S8zc21EthjpCg7ckslAD4PfovWyErQ0p' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1S8zc21EthjpCg7ckslAD4PfovWyErQ0p" -O dataset_public.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./dataset_public.zip -d dataset_public

# Remove the downloaded zip file
rm ./dataset_public.zip
