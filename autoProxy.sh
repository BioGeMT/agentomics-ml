#!/bin/bash

cp Dockerfile Dockerfile.bak

for proxy in $(echo "http_proxy HTTP_PROXY https_proxy HTTPS_PROXY");do
  value=$(printenv $proxy)
  status=$?
  if [ $status -eq 0 ];then
    echo "ENV ${proxy} ${!proxy}" >> temp.txt
    echo "RUN export ${proxy}=\$${proxy}" >> temp.txt
  fi
done

cat temp.txt > Dockerfile.tmp
echo "" >> Dockerfile.tmp
echo "RUN conda config --set ssl_verify false" >> Dockerfile.tmp
echo "" >> Dockerfile.tmp
cat Dockerfile >> Dockerfile.tmp

mv Dockerfile.tmp Dockerfile

rm temp.txt
