import struct, os  
import numpy as np
# get the object index in multi-object scene.

def pcd2npy(file_number):
    last_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))     
    formal_pcd_path = os.path.join(last_path, "bullet3/my_pybullet_code/point/", str(file_number) + "00000.pcd")
    formal_point = open(formal_pcd_path, 'r')

    pcd_path = os.path.join(last_path, "bullet3/my_pybullet_code/point_color/", str(file_number) + "00000.pcd")
    pcd_npy_path = pcd_path.replace("pcd", "npy")
    pcd_npy_path = pcd_npy_path.replace("point_color", "point_npy")

    
    formal_point_txt = formal_point.readlines()
    point_new = np.full([len(formal_point_txt)-11, 6], -1.0, dtype=np.float32)

    for i in range(11, len(formal_point_txt)):
        line = formal_point_txt[i]
        f_line = formal_point_txt[i].split(" ")

        f_rgb = f_line[-2]
        fr = (int(float(f_rgb)) >> 16)&0x0000ff
        fg = (int(float(f_rgb)) >> 8)&0x0000ff
        fb = (int(float(f_rgb))) & 0x0000ff

        point_new[i-11,0] = float(f_line[0])
        point_new[i-11,1] = float(f_line[1])
        point_new[i-11,2] = float(f_line[2])
        point_new[i-11,3] = fr
        point_new[i-11,4] = fg
        point_new[i-11,5] = fb
    print(point_new)
    np.save(pcd_npy_path, point_new)


def save_ply(vertices, filename):
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = \
            '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def get_index(file_number):
    #last_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))   
    last_path = os.path.abspath(os.getcwd())    
    formal_pcd_path = os.path.join(last_path, "bullet3/my_pybullet_code/point/", str(file_number) + "00000.pcd")
    pcd_path = os.path.join(last_path, "bullet3/my_pybullet_code/point_color/", str(file_number) + "00000.pcd")
    pcd_npy_path = pcd_path.replace("pcd", "npy")
    pcd_npy_path = pcd_npy_path.replace("point_color", "point_npy")
    pos_path = os.path.join(last_path, "bullet3/my_pybullet_code/pos/", str(file_number) + ".txt")
    formal_point = open(formal_pcd_path, 'r')

    f = open(pcd_path, 'r')
    pos = open(pos_path, 'r')
    #f_npy = open(pcd_npy_path, 'r')
    txt = f.readlines()
    pos_txt = pos.readlines()
    formal_point_txt = formal_point.readlines()
    point_new = np.full([len(txt)-11, 7], -1.0, dtype=np.float32)

    color_r, color_g, color_b = [], [], []
    for i in range(11, len(txt)):
        line = txt[i]
        #print(line)
        rgb = line.split(" ")[-2]
        r = (int(float(rgb)) >> 16)&0x0000ff
        g = (int(float(rgb)) >> 8)&0x0000ff
        b = (int(float(rgb))) & 0x0000ff
        if r != 255 and r != g:
            #print(r,g,b)
            if r not in color_r:
                color_r.append(r)
                color_g.append(g)
                color_b.append(b)
    color = color_r
    color.sort()

    index = []
    for i in range(len(pos_txt)):
        if i % 2 == 0:
            line = pos_txt[i]
            index.append(int(line.split("_")[0]))
    index.sort()
    assert len(index) >= len(color)

    if len(index) > len(color):
        check_num = len(index) - len(color)
        print("There is/are", check_num, "object(s) blocked")
        color_time = 0.01 # set in sim.py
        color_time *= 255 # color range 0-255
        compute_time = np.zeros((check_num+1, len(color)))
        for i in range(check_num+1):
            compute_time[i] = np.array(color) / np.array(index[i:len(color)+i])
        compute_time -= color_time
        compute_time = np.abs(compute_time)
        compute_time_min = np.min(compute_time, axis=0)
        compute_time_min_index = np.argmin(compute_time, axis=0)
        correspond_index = np.arange(len(color)) 
        correspond_index += compute_time_min_index
        index = list(np.array(index)[correspond_index])
    assert len(index) == len(color)

    for i in range(11, len(txt)):
        line = txt[i]
        rgb = line.split(" ")[-2]
        r = (int(float(rgb)) >> 16)&0x0000ff
        g = (int(float(rgb)) >> 8)&0x0000ff
        b = (int(float(rgb))) & 0x0000ff

        f_line = formal_point_txt[i].split(" ")
        f_rgb = f_line[-2]
        fr = (int(float(f_rgb)) >> 16)&0x0000ff
        fg = (int(float(f_rgb)) >> 8)&0x0000ff
        fb = (int(float(f_rgb))) & 0x0000ff

        point_new[i-11,0] = float(f_line[0])
        point_new[i-11,1] = float(f_line[1])
        point_new[i-11,2] = float(f_line[2])
        point_new[i-11,3] = fr
        point_new[i-11,4] = fg
        point_new[i-11,5] = fb
        for j in range(len(color)):
            if r == color[j]:
                point_new[i-11,6] = index[j]
                break
    
    index = np.nonzero((point_new[:,0]>-0.4) & (point_new[:,0]<0.4))
    point_new = point_new[index]
    point_new = point_new[~np.isnan(point_new[:,0])]
    np.save(pcd_npy_path, point_new)
    save_ply(point_new[:, :6], pcd_npy_path.replace("npy", "ply"))


for i in range(1000):
    print("----------------------The ", i+1, "th file------------------------------")
    #pcd2npy(i)
    get_index(i)
