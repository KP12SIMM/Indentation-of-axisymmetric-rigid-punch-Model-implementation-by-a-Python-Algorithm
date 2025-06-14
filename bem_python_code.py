
Created on 08/10/2020

@author: krupal Patel & Etienne Barthel
"""

from math import pi,exp,floor
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from numpy import linalg as LA
from decimal import Decimal
############################################################################
#Indentation free param
############################################################################
# contact radius
a=1.01
#Layer thickness
T=0.5
###0#######################################################################
#Tip parameters
############################################################################
#Radius for the sphere or the flat punch
R=10
#tan(w) of the cone
tanw=2.8
############################################################################
#Bilayer real parameters
############################################################################
#Young's modulus & Poisson's ratio of the half-space (Mpa)
E0=10    #2950.81    #2900
v0=0.4  #0.4754      #0.45
#Young's modulus & Poisson's ratio of the layer (MPa)
E1=1     #2.999#2.999  #0.003
v1=0.25  #0.4999#0.4999#0.4999 # 0.5
############################################################################
#Reduced Normalized variables
############################################################################
tau=T/a # Normalised thickness
############################################################################
#numeric
############################################################################
#System size - typical 1000 - increase at large a/T
n=1000
#FFT cut off B/2 and points number 2^k
B=700    # this is cVmax in the original routine
vT=20   # The number of points in the fourier space
############################################################################
#Tables
############################################################################
G=np.array([0,1])
ZF=[]
M=np.eye(n)
V=np.array([0,1])
X=np.array([0,1])
#Fonction Z du papier
def GreenNormal(kt):
    gamma1=3-4*v1
    gamma3=3-4*v0
# difference HERE
    alpha=E1*(1+v0)/(E0*(1+v1))
    A=(alpha*gamma3-gamma1)/(1+alpha*gamma3)
    B=(alpha-1)/(alpha+gamma1)
    denominator=1-(A+B+4*B*(kt**2))*exp(-2*kt)+A*B*exp(-4*kt)
    C=(1+4*B*kt*exp(-2*kt)-A*B*exp(-4*kt))/denominator
    return C-1

#A smooth periodic approx
def makeZforFFT():
    global vMax
    vMax=float(B)/tau
    l=2**vT
    table=[1.2]*l

    dx=float(vMax)/(l-1)
    for i in range(int(l/2)):
        x=i*(dx)
        val=GreenNormal(x*tau)
        table[i]=val
        table[-i-1]=val
#    plt.plot([i*dx for i in range(l)], table)
    return table

#cosine transform
def makeTFC():
    global ZF
    Tinput=makeZforFFT()
    T_FFT=np.fft.rfft(Tinput)
    l=2**vT
#    print 1./l*vMax
    ZF=[x/2*vMax/l for x in T_FFT.real]
    #np.savetxt('result.txt',ZF)
#generates the coeficients of linear system
def generateMatrixTerms():
    global M
    makeTFC()
    bufferM=np.eye(n+1)
#    plt.figure("oh")
#    plt.plot([2*pi*y/vMax for y in range(len(ZF))],ZF)
    for i in range(n+1):
        for j in range(1,n):
            s=float(i)/n
            r=float(j)/n
            index_r_moins_s=abs(int((r-s)*vMax/(2*pi)))
            index_r_plus_s=abs(int((r+s)*vMax/(2*pi)))

#        print 'ind i, ind j, ' +str(len(ZF))+'    ' +str(vMax/(2*pi))+'    '+str(index_r_moins_s)+ '    '+ str(index_r_plus_s)+'    '+ str(i)+'    '+ str(j)
            bufferM[i,j]+=(ZF[index_r_moins_s]+ZF[index_r_plus_s])/(n*pi)
#            print str(bufferM[i,j])
        j=0
        s=float(i)/n
        r=float(j)/n
        index_r_moins_s=abs(int((r-s)*vMax/(2*pi)))
        index_r_plus_s=abs(int((r+s)*vMax/(2*pi)))
        bufferM[i,0]+=(ZF[index_r_moins_s]+ZF[index_r_plus_s])/(2*n*pi)
        #bufferM[i,0]/=1.

        if shape=='FLAT':
            j=n
            s=float(i)/n
            r=float(j)/n
            index_r_moins_s=abs(int((r-s)*vMax/(2*pi)))
            index_r_plus_s=abs(int((r+s)*vMax/(2*pi)))
            bufferM[i,n]+=(ZF[index_r_moins_s]+ZF[index_r_plus_s])/(2*n*pi)
            #bufferM[i,n]/=1
        else:
            bufferM[i,n]=-1.
    M=bufferM

#Right hand vector
def generateRightVector():
    global V
    Ar=[0]*(n+1)
    if shape=='CONIC':
        for k in range(n+1):
            Ar[k]=-float(k)/n
    elif shape=='SPHERE':
        for k in range(n+1):
            Ar[k]=-(float(k)/n)**2
    elif shape=='FLAT':
        for k in range(n+1):
            Ar[k]=1
    V=Ar


#Let's go!
def runFEBM():
    global X,G,DELTA,PI,delta,P,G_fp,S_fp
    print ('shape:%s, contact radius : %5.2f, radius ? : %5.2f, sphere radius : %5.2f, tan(w) : %5.2f, E0 : %5.2f, E1 : %5.2f, nu0 : %5.2f, nu1 : %5.2f' % (shape,a,tau*a,R,tanw,E0,E1,v0,v1))
    CON=LA.cond(M)
    print(CON)
    generateMatrixTerms()
    generateRightVector()

   # X=np.linalg.solve(M,V)

    LU=linalg.lu_factor(M)

    X=linalg.lu_solve(LU,V)

    G=np.array([0.0]*(n+1))
    for k in range(n):
        G[k]=X[k]
    PI=4./n*np.sum(G)  # Normalized force PI

    G_rho=X[n]

    if shape == 'CONIC':
        delta=(pi/2)*(a/tanw)*G_rho  # penetration
        P=(pi/4)*(a**2*(E1/(1-v1**2))/tanw)*PI # force
        e_eq=PI/(2*G_rho**2)
        E_eq=E1*e_eq/(1-v1**2)
        print (G_rho,PI,delta,P,E_eq)
    elif shape=='SPHERE':
        delta=(a**2/R)*G_rho # penetration
        P=0.5*(a**3*(E1/(1-v1**2))/R)*PI # force
        e_eq=(3./8.)*PI/(G_rho**(1.5))
        E_eq=E1*e_eq/(1-v1**2)
        print (G_rho,PI,delta,P,E_eq)
    elif shape=='FLAT':
        delta=0.001
        #G_fp=(G_rho*2*(1-v1**2))/(E1*delta)
        G_fp=(G_rho*E1*delta)/((1-v1**2)*(np.pi*a)**(1/2))    # Stress Intensity
        #G_fp=(delta*E1)/((1-v1**2)*(2*(2*np.pi*a)**(1/2)))
        ##### G_fp=(delta*E1)/((1-v1**2)*((np.pi*a)**(1/2))) # stress intensity factor
        S_fp=PI*E1/(1-v1**2)/2*a# stiffness
        e_eq=PI/4.
        E_eq=E1*e_eq/(1-v1**2) # Equivalent modulus of the system of halfsace and layer

        print ('G : %5.2f, penetration : %5.2g, force : %5.2f, G : %5.2g, stiffness : %5.2f, equiv. modulus : %5.2f ' % (G_rho,delta,PI/4.0,G_fp,S_fp,E_eq) )

# shape='CONIC'
# runFEBM()
# shape='SPHERE'
# runFEBM()
shape='FLAT'
runFEBM()
