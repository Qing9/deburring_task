import os
import cv2
import numpy as np
import math
import random
from util import *

ban_list=[45]

output_path='../../data/synthetic images/'
dirs=['有毛刺/','无毛刺/']
img_size=256
# tf_size=276

class ProcessFunc():
    def __init__(self,):
        super(ProcessFunc,self).__init__()
    @staticmethod
    def sin1(x):
        return 1.7*math.sin(0.17*x)

    @staticmethod
    def sin2(x):
        return 1.2*math.sin(0.1*x)
    @staticmethod
    def sin3(x):
        return 2*math.sin(0.05*x)
    @staticmethod
    def sin4(x):
        return 2*math.sin(0.1*x)
    @staticmethod
    def protrusion(a,shift,x):
        return (x/a)**2+shift
    @staticmethod
    def depressed(a,shift,x):
        return a*(x-shift)**2


def normal_padding(img,h,w):

    for row in range(h):
        start_pad=False
        for col in range(w):
            if col+1<w and img[row,col]==255 and start_pad==False and img[row,col+1]==0:
                start_pad=True
            elif start_pad==True and img[row,col]==0:
                img[row,col]=255
            elif start_pad==True and img[row,col]==255:
                start_pad=False
    return img
def horizon_padding(img,bs0,cols0,bs1,cols1,row_start):
    l=len(cols0)
    for i in range(l):
        img[row_start+i,bs0-cols0[i]:bs1-cols1[i]]=255
    return img

def vertical_padding(img,bs0,rows0,bs1,rows1,col_start):
    l = len(rows0)
    for i in range(l):
        img[bs0-rows0[i]:bs1-rows1[i],i+col_start]=255

        img[bs1-rows1[i]+1:bs1,i+col_start]=0
    return img
def protrusion_or_depressed(vl,vh):
    res=[]
    max_pos = round(random.uniform(0.2, 0.8) * vl)
    a = vh / max_pos ** 2
    for x in range(0, max_pos + 1):
        y = round(ProcessFunc.depressed(a, 0, x))
        res.append(y)

    b = (vh / (vl- max_pos + 1) ** 2)
    for x in range(max_pos + 1, vl):
        y = round(ProcessFunc.depressed(b, vl, x))
        res.append(y)
    return res
def process_x_list(total_vl):
    y_list=[0 for i in range(total_vl)]

    # total_vl=end-start
    end=round(0.95*total_vl)
    # print(start,total_vl,end)
    start=round(0.1*total_vl)
    alpha = random.uniform(0.05, 0.1)
    mode_list=['sin','skip','protrusion','depressed']
    mode= random.choice(mode_list)
    while start + alpha * total_vl <=end:

        vl=round(alpha * total_vl)
        # print(start,vl,alpha)
        if mode == 'sin':
            func = random.choice(['sin1', 'sin2', 'sin3', 'sin4'])
            sin = getattr(ProcessFunc, func)
            for x in range(start,start+vl):
                y = round(sin(x))
                y_list[x]=y

        elif mode=='protrusion':
            vh=random.randint(2,5)
            res=protrusion_or_depressed(vl,vh)
            for x in range(start,start+vl):
                y_list[x]=res[x-start]
        elif mode=='depressed':
            vh=random.randint(2,3)
            res=protrusion_or_depressed(vl,vh)
            for x in range(start,start+vl):
                y_list[x]=-res[x-start]
        #check
        if y_list[start]>=2:
            # print('haha',y_list[start],start)
            for i in range(1,y_list[start]):
                y_list[start-i]=y_list[start-i+1]-1
        start += vl
        times=0
        if y_list[start-1] >= 2:
            # print('end', y_list[start-1], start-1)
            times = y_list[start - 1]
            for i in range(1, times):
                y_list[start-1+i]=y_list[start-2+i]-1
        start+=times

        mode = random.choice(mode_list)

        if mode == 'sin':
            alpha = random.uniform(0.2, 0.3)
        elif mode=='protrusion':
            alpha = random.uniform(0.2, 0.4)
        elif mode=='depressed':
            alpha = random.uniform(0.1,0.2)
        else:
            alpha=random.uniform(0.1,0.25)
    return y_list

def draw_edge(img,areas,type):
    deveining_edge_img=np.zeros((img_size,img_size))
    y_list_res=[]
    for index,area in enumerate(areas):
        start=area[0]
        end=area[1]
        baseline=area[2]

        y_list=process_x_list(end-start)
        y_list_res.append(y_list)
        if type[index]=='vertical':
            deveining_edge_img[start:end, baseline] = 255
            for x in range(start,end):
                img[x,baseline-y_list[x-start]]=255

        else:
            deveining_edge_img[baseline,start:end+1]=255
            for x in range(start,end):
                img[baseline-y_list[x-start],x]=255

    return img,deveining_edge_img,y_list_res
def workpiece1(img):
    areas=[(0,200,13),(0,100,33),(120,180,33),(0,100,223),(120,180,223),(0,200,243),(33,223,100),(33,223,120),(33,223,180),(13,243,200)]
    type=['vertical','vertical','vertical','vertical','vertical','vertical','horizontal','horizontal','horizontal','horizontal']
    veining_edge_img,deveining_edge_img,y_list_res=draw_edge(img,areas,type)
    deveining_img=normal_padding(deveining_edge_img.copy(),200,243)

    border1=[]
    border1+=y_list_res[1]
    border1+=[0,]*20
    border1+=y_list_res[2]
    border1+=[0,]*20
    veining_img=horizon_padding(veining_edge_img.copy(),13,y_list_res[0],33,border1,row_start=0)
    border1=[]
    border1+=y_list_res[3]
    border1+=[0,]*20
    border1+=y_list_res[4]
    border1+=[0,]*20
    veining_img = horizon_padding(veining_img, 223, border1, 243, y_list_res[5],row_start=0)
    veining_img=vertical_padding(veining_img,bs0=100,rows0=y_list_res[-4],bs1=120,rows1=y_list_res[-3],col_start=33)
    border0=[0,]*20
    border0+=y_list_res[-2]
    border0+=[0,]*20
    veining_img = vertical_padding(veining_img, bs0=180, rows0=border0, bs1=200, rows1=y_list_res[-1],col_start=13)

    return veining_edge_img,deveining_edge_img,veining_img,deveining_img

def workpiece2(img):
    areas=[(0,256,128),(0,256,153),]
    type=['vertical','vertical']
    veining_edge_img,deveining_edge_img,y_list_res=draw_edge(img,areas,type)
    deveining_img = normal_padding(deveining_edge_img.copy(), 256, 153)
    veining_img=horizon_padding(veining_edge_img.copy(),bs0=128,cols0=y_list_res[0],bs1=153,cols1=y_list_res[1],row_start=0)

    return veining_edge_img,deveining_edge_img,veining_img,deveining_img

def workpiece3(img):
    areas=[(40,256,90),(65,256,115),(90,256,40),(115,256,65)]
    type=['vertical','vertical','horizontal','horizontal']
    veining_edge_img, deveining_edge_img, y_list_res = draw_edge(img, areas, type)
    deveining_img = normal_padding(deveining_edge_img.copy(), 256, 256)
    veining_img=vertical_padding(veining_edge_img.copy(),bs0=40,rows0=y_list_res[2],bs1=65,rows1=[0,]*25+y_list_res[-1],col_start=90)
    veining_img = horizon_padding(veining_img, bs0=90, cols0=y_list_res[0], bs1=115, cols1=[0,]*25+y_list_res[1],
                                  row_start=40)


    return veining_edge_img, deveining_edge_img, veining_img, deveining_img

def workpiece4(img):
    areas=[(0,256,230)]
    type = ['horizontal']
    veining_edge_img, deveining_edge_img, y_list_res = draw_edge(img, areas, type)
    deveining_img = deveining_edge_img.copy()
    deveining_img[230:,:]=255
    veining_img=veining_edge_img.copy()
    for col in range(0,256):
        for row in range(220,256):
            if veining_img[row,col]==255:
                veining_img[row:,col]=255
                break
    return veining_edge_img, deveining_edge_img, veining_img, deveining_img


def gen_data():
    workpieces=[workpiece1]

    tf_list=[False,True,True,True,True]

    tf_angle_list=[0,0,90,180,270]
    shift_list=[0]
    random_angle_list=[0,]
    save_path_list=['edge/有毛刺/','edge/无毛刺/','padding/有毛刺/','padding/无毛刺/']
    for dir in save_path_list:
        if not os.path.exists(output_path+dir):
            os.makedirs(output_path+dir)
    for i,workpiece in enumerate(workpieces):

        for num in range(0*10,1*10):
            fn='{}.jpg'.format(num)
            zero_img=np.zeros((img_size,img_size))
            veining_edge_img,deveining_edge_img,veining_img,deveining_img=workpiece(zero_img)
            img_list=[veining_edge_img,deveining_edge_img,veining_img,deveining_img]
            random_angle=random.uniform(-random_angle_list[i],random_angle_list[i])
            shift=random.uniform(0,shift_list[i])
            translation_matrix = np.float32([[1, 0, shift], [0, 1, shift]])
            tf=tf_list[num%5]
            for img,dir in zip(img_list,save_path_list):

                img = cv2.warpAffine(img, translation_matrix, (img_size, img_size))
                if tf==True:

                    rotation_matrix = cv2.getRotationMatrix2D((img_size//2,img_size//2), random_angle, 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (img_size,img_size))

                    rotation_matrix = cv2.getRotationMatrix2D((img_size//2,img_size//2), tf_angle_list[num%5],1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (img_size, img_size))

                # else:
                #
                #     img = cv2.warpAffine(img, translation_matrix, (img_size, img_size))
                imsave(img,output_path+dir+fn)

            # affine_transform_img[gap:gap+img_size,gap:gap+img_size]=veining_img
            # 计算平移矩阵
            # translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
            # veining_img=cv2.warpAffine(veining_img, translation_matrix, (img_size, img_size))
        break



make_path(output_path)
gen_data()
# save_path_list=['edge/有毛刺/','edge/无毛刺/','padding/有毛刺/','padding/无毛刺/']
# for dir in save_path_list:
#     print(len(os.listdir(output_path+dir)))
