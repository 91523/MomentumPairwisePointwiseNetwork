# bin/bash

BASE_PATH="data"
FILE="ml-100k.zip"

if [ ! -f $BASE_PATH ]; then
  mkdir $BASE_PATH
fi

# download data
if [ ! -f $BASE_PATH/$FILE ]; then
  echo "Start downloading file $FILE ..."
  wget -O $BASE_PATH/$FILE http://files.grouplens.org/datasets/movielens/ml-100k.zip
else
  echo "File already exists, skip downloading."
fi

# decompression
cd $BASE_PATH && unzip $FILE