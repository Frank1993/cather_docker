tar -xvf /openmpi-4.0.0.tar.gz -C /
ls /
cd /openmpi-4.0.0/
sh configure --prefix=/usr/local
make all install
ldconfig