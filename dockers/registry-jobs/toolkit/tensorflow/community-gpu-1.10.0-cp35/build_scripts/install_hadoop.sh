#!/usr/bin/env bash
# Install Hadoop

HADOOP_VERSION="2.7.6"

set +e
if [[ ! -f /usr/local/hadoop-${HADOOP_VERSION}/bin/hadoop ]]; then
  set -e
  aria2c -k 1M -x 5 -j 5 --quiet --http-accept-gzip=true http://www-us.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz
  tar xzf hadoop-${HADOOP_VERSION}.tar.gz -C /usr/local
fi
ln -sf /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop
rm -rf /usr/local/hadoop/etc/hadoop
cp -r /root/build_scripts/hadoop /usr/local/hadoop/etc/hadoop
