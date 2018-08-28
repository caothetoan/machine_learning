git clone https://github.com/davisking/dlib.git
cd dlib
brew install cmake

mkdir build;
cd build/
cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1
cmake --build  .

pip install face_recognition
