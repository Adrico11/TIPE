# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:39:46 2020

@author: snice
"""

#Importation des modules
import random
from math import log,sqrt,cos,sin,pi,acos,floor,exp
import numpy as np
from PIL import Image, ImageDraw  
import matplotlib.pyplot as plt
from time import *


## Modèle 1 Continu

l0=1064*10**(-9) #longueur d'onde
c=3.00*10**8
h=6.63*10**(-34)



R=[5.00,12.5,25.0] #Rainfall rate
#rho=[0.2,0.07]
N=10**4
E=N*h*c/l0   
Ar=0.2
Cs=c*E*Ar*0.5


def courberainfall(N,rho):
    X=np.linspace(0.2,5,100) #valeurs de L
    R=[5,12.5,25] #valeurs de précipitation
    for j in R:
        Y=[]
        for i in X:
            P=((rho/(i)**2))*exp(-0.8*(j**0.6)*i)/100
            print(i)
            print(P)
            Y.append(P)   
        plt.plot(X,Y,label=str(j)+"mm/h")     
    plt.legend()
    plt.show()
    
    
    

## Modèle 2 Discret probabiliste


l0=1064*10**(-9) #longueur d'onde
c=3.00*10**8
h=6.63*10**(-34)
h0=h/(2*pi)

#Coefficients d'atténuation en scattering et attenuation (m-1)
R=[5.00,12.5,25.0,100]
Coeffs=[(0.0013,0.00132),(0.0024,0.00244),(0.0038,0.00387),(0.0097,0.00991)]

def HG(g,k):
   return (acos((1/(2*g))*((1+g**2)-((1-g**2)**2)/((1-g+2*g*k)**2)))) 
   
def transfo(Ux,Uy,Uz,a,b):
    if abs(Uz)<0.99999:
        Ux=(sin(a)*(Ux*Uz*cos(b)-Uy*sin(b))/((1-Uz**2)**0.5))+Ux*cos(a)
        Uy=(sin(a)*(Uy*Uz*cos(b)-Ux*sin(b))/((1-Uz**2)**0.5))+Uy*cos(a)
        Uz=-sin(a)*cos(b)*((1-Uz**2)**0.5)+Uz*cos(a)
    else:
        Ux=sin(a)*cos(b)
        Uy=sin(a)*sin(b)
        Uz=np.sign(Uz)*cos(a)
    return (Ux,Uy,Uz)    
       

def positionecran(L,x0,y0,z0,x1,y1,z1):
    if z1!=z0:
        x=x0+(L-z0)*((x1-x0)/(z1-z0))
        y=y0+(L-z0)*((y1-y0)/(z1-z0))
        return (x,y)

#position cible (L=position selon z,Dx=largeur selon x,Dy=largeur selon y)

def propagation1(N,L,Dx,Dy,g,Ma,Ms):
    W1=[]
    X1=[]
    Y1=[]
    I1=[]
    d1=[]
    for i in range(N):
        j=0
        w=1 #énergie du photon au départ
        (x,y,z)=(0,0,0)
        (Ux,Uy,Uz)=(0,0,1)
        d=0
        while j<1000 and z<L and w>0:
            k=random.uniform(1,0)
            b=random.uniform(0,2*pi)
            a=HG(g,k)
            l=(-log(k))/(Ma+Ms)
            (Ux,Uy,Uz)=transfo(Ux,Uy,Uz,a,b)
            (x0,y0,z0)=(x,y,z)
            (x,y,z)=(x+l*Ux,y+l*Uy,z+l*Uz)
            w=w*(1-(Ms/(Ms+Ma)))
            d+=l
            j+=1
        (x,y)=positionecran(L,x0,y0,z0,x,y,z)
        if abs(x)<=Dx/2 and abs(y)<=Dy/2 and w>0 and z>=L: #le photon a atteint la cible
            W1.append(w)
            X1.append(x)
            Y1.append(y)
            I1.append([Ux,Uy,Uz])
            d1.append(d)
    n1=len(X1)        
    return (W1,X1,Y1,I1,d1,n1)   
    
def reflexion(Ux,Uy,Uz):
    return (Ux,Uy,-Uz)
        

def propagation2(L,Ex,Ey,g,W1,X1,Y1,I1,d1,n1,Ma,Ms):
    n2=0
    W2=[0]*n1
    X2=[0]*n1
    Y2=[0]*n1
    I2=[0]*n1
    d2=[0]*n1
    for i in range(n1):
        j=0
        w=W1[i]
        (x,y,z)=(X1[i],Y1[i],0)
        (Ux,Uy,Uz)=(I1[i][0],I1[i][1],I1[i][2])
        d=0
        while j<1000 and z<L and w>0:
            k=random.uniform(1,0)
            b=random.uniform(0,2*pi)
            a=HG(g,k)
            l=(-log(k))/(Ma+Ms)
            (Ux,Uy,Uz)=transfo(Ux,Uy,Uz,a,b)
            (x0,y0,z0)=(x,y,z)
            (x,y,z)=(x+l*Ux,y+l*Uy,z+l*Uz)
            w=w*(1-(Ms/(Ms+Ma)))
            d+=l
            j+=1
        (x,y)=positionecran(L,x0,y0,z0,x,y,z)   
        if abs(x)<=Ex/2 and abs(y)<=Ey/2 and w>0 and z>=L: #le photon a atteint le capteur
            W2[i]=w
            X2[i]=x
            Y2[i]=y
            I2[i]=(Ux,Uy,Uz)
            d2[i]=d+d1[i]
            n2+=1
    return (W2,X2,Y2,I2,d2,n2)   
    

def propagationtot(N,L,Dx,Dy,g,Ex,Ey):
    return (propagation2(L,Ex,Ey,g,propagation1(N,L,Dx,Dy,g)))        
        
def cadrage(m,M):
    if m!=M:
        return (1/(M-m),-m/(M-m))
    else:
        return(1,0)        
        

def photonsecran(W1,X1,Y1,Dx,Dy,N,d1,d2,t,W2):
    for i in range (len(W1)):
            (c,d)=cadrage(min(W1),max(W1))
            w0=int((c*W1[i]+d)*(len(RGB1)-1))
            print(w0,W1[i])
            plt.scatter(X1,Y1,s=15)
    plt.scatter(0,0,s=15,c="red")
    T=tempstot(d1,d2)
    P=puissance(N,t,T,W2)
    axes=plt.gca()
    axes.set_xlim([-Dx/2,Dx/2])
    axes.set_ylim([-Dy/2,Dy/2])
    plt.title(str(len(W1))+" photons sur la cible"+"("+str(N)+")"+"\n"+"Puissance relative reçue = "+str(float(P)))
    plt.legend()
    plt.show()
                   
    
RGB1=[(0,0,255),(50,205,50),(255,255,0),(255,0,0),(148,0,211)]

def tempstot(d1,d2):
    D=[]
    for i in range (len(d1)):
        if d2[i]!=d1[i]: #sinon le photon n'a pas atteint le capteur
            D.append(d2[i])
    T=[x/c for x in D]
    return T


def Temps(T):
    A=len(T)
    T1=max(T)
    while len([x for x in T if x>=T1])>=0.4*A:
        T1=T1-A/sum(T)
    T2=min(T)   
    while len([x for x in T if x<=T2])>=0.4*A:
        T2=T2-A/sum(T)    
    return (T1,T2)    
         

def puissance(N,t,T,W2):
    if T!=[]:
        (Tmax,Tmin)=(max(T),min(T))
        P1=(N*h*c)/(t*l0) 
        P2=0
        if Tmax!=Tmin: #il y a plus de deux photons qui arrivent sur le capteur
            E=sum(W2)*h*c/l0
            P2=E/(Tmax-Tmin)
        return (P2/P1)
    return (None)
    
def puissancemoy(N,L,g,t,Dx,Dy,Ex,Ey,M,Ma,Ms):
    P=[]
    for i in range(M):
        (W1,X1,Y1,I1,d1,n1)=propagation1(N,L,Dx,Dy,g,Ma,Ms) 
        (W2,X2,Y2,I2,d2,n2)=propagation2(L,Ex,Ey,g,W1,X1,Y1,I1,d1,n1,Ma,Ms)   
        T=tempstot(d1,d2)
        p=puissance(N,t,T,W2)
        if p!=None and p>0 :
            P.append(p)
    if P!=[]:        
        p0=sum(P)/len(P)      
        return (p0)  
     

def courbefinale(N,g,t,Ex,Ey,M,Ma,Ms):
    "Puissance relative reçue en fonction de L pour différentes surfaces de la cible"
    X=np.linspace(0.5,100,50) #valeurs de L
    Z=[0.5,5,20] #valeurs de la surface de la cible
    for j in range (len(Z)):
        Y=[]
        for i in range (len(X)):
            p0=puissancemoy(N,X[i],g,t,(Z[j])**0.5,(Z[j])**0.5,Ex,Ey,M,Ma,Ms)
            if p0!=None:
                Y.append(p0)
            else:
                Y.append(0)    
        plt.plot(X,Y,label=str(Z[j])+"m²")
    plt.legend()     
    plt.show() 

    

def courbefinale2(N,g,t,Ex,Ey,Dx,Dy,M):
    "Puissance relative reçue en fonction de L pour différentes précipitations"
    X=np.linspace(0.5,10,50)
    R=[5.00,12.5,25]
    Coeffs=[(0.0013,0.00132),(0.0024,0.00244),(0.0038,0.00387)]
    for j in range (len(Coeffs)):
        Y=[]
        for i in range (len(X)):
            p0=puissancemoy(N,X[i],g,t,Dx,Dy,Ex,Ey,M,Coeffs[j][0],Coeffs[j][1])
            if p0!=None:
                Y.append(p0)
            else:
                Y.append(0)    
        plt.plot(X,Y,label=str(R[j])+"mm/h")
    plt.legend()
    plt.show() 