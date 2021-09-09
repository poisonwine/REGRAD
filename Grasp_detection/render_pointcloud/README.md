
- [Render Point Cloud](#Render-Point-Cloud)
  - [Installation](#Installation)   
    - [pybullet](#pybullet)  
    - [blender & blensor](#blender--blensorhttpswwwblensororgpagesdownloadshtml)   
    - [other tips](#other-tips)      
      - [generate able of contents in Markdown](#generate-able-of-contents-in-markdown)
  - [Code](#Code) 
    - [rendering point cloud used blensor](#rendering-point-cloud-used-blensor)  
- [REGNet](#REGNet)
  - [Installation](#Installation)   
    - [python-pcl](#python-pcl)  
      - [pcl](#pcl)  
      - [python-pcl](#python-pcl)  

---
# Render Point Cloud
## Installation
### pybullet
```
git clone https://github.com/bulletphysics/bullet3.git
cd bullet3 
python setup.py build
sudo python setup.py install
```
And then we can ```cd examples/pybullet/examples``` and run python scripts

---

### [blender & blensor](https://www.blensor.org/pages/downloads.html)

Download the blensor from: https://www.blensor.org/dload/blensor_1.0.18-RC10_x64.tbz

After unzip, we can directly run ```./blender```

[Blender Python API Documentation](https://docs.blender.org/api/current/)

```
cd blensor
./blender --python xxx.py
```
In the blensor of my laptop, I have changed the rendering code of kinect (remove the struct.unpack and pack in 2.74/scripts/addons/blensor/evd.py #130) -> to generate pointcloud with masks

---

### other tips

#### generate able of contents in Markdown
1. pandoc
- download the [pandoc](https://github.com/jgm/pandoc/releases/download/2.10/pandoc-2.10-1-amd64.deb)
- `sudo dpkg -i pandoc-2.10-1-amd64.deb`
- `pandoc -s --toc --toc-depth=4 README.md -o README.md`

2. VSCode TOC Plugin

---
## Code
### rendering point cloud used blensor
[Blender reference manual](https://docs.blender.org/manual/en/latest/)

samples:
1. **scripts/startup/bl_ui** for the user interface
2. **scripts/startup/bl_operators** for operators


---
# REGNet
## Installation
### python-pcl
#### pcl
```
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
git checkout -b pcl-1.9.1 pcl-1.9.1 (tag->branch)
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```

- ERROR:&nbsp; &nbsp; /usr/lib/../lib64/libSM.so: undefined reference to `uuid_unparse_lower@UUID_1.0' \
SOLVED:&nbsp; ```conda install -c conda-forge xorg-libsm```

#### python-pcl
```
git clone https://github.com/strawlab/python-pcl.git
cd python-pcl
python setup.py build_ext -i
python setup.py install
```