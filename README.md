2️1. Install Python 3.10 (if not available)

sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

cd /tmp
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xvzf Python-3.10.14.tgz
cd Python-3.10.14
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
 
 2. Create and Activate Virtual Environment

python3.10 -m venv cam
source cam/bin/activate






✅ requirements

face-recognition==1.3.0
face_recognition_models==0.3.0
opencv-python==4.11.0.86
numpy==1.24.4
pillow==9.5.0
