import math
import numpy as np
import matplotlib.pyplot as plt
# second commit
# thrid commit trying with pycharm
# 4th commmit
# given remote repo is different now.
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def f(u):
    return  u ** 2 / (u ** 2 + 0.1 * (1 - u) ** 2)

def sigmoid_steep(x):
  return 1 / (1 + np.exp(-x * 1e5))
u = np.linspace(0,1,80)




def F_sigmoid(u, k):
    x = 0.3182
    y = 0.7955
    S = (u-x)/(y-x)
    #step_fun = sigmoid_steep(S)-sigmoid_steep(S-1)
    step_fun = sigmoid(S*k)-sigmoid(k * (S-1))
    FF = x * (1 - f(S) ) + y* (f(S))
    return (step_fun * FF  + (1- step_fun) * u )



def F_normal(u):
    x = 0.3182
    y = 0.7955
    FF = np.linspace(0, 1, 64)
    S = (u-x)/(y-x)
    for i in range(len(S)):
        if 0<S[i]<1:
            FF[i] = x * (1 - f(S[i]) ) + y* (f(S[i]))
        else:
            FF[i]= u[i]
    return FF

def F_refine(u):
    x = 0.3182
    y = 0.7955
    S = (u-x)/(y-x)
    step_fun = sigmoid(S)-sigmoid(S-1)
    FF = x* (1 - f(S) ) + y* (f(S))
    return (step_fun * FF  + (1- step_fun) * u )

#plt.plot(uu)
xx = np.loadtxt('file.csv',  dtype= 'float16' ,delimiter=",")
uu = np.linspace(0,1,64)
B = F_normal(uu)
dB = (B[1:]- B[:-1])/max(np.diff(uu))

B_sigmoid = F_sigmoid(uu, 10)
dB_sig = (B_sigmoid[1:]- B_sigmoid[:-1])/max(np.diff(uu))
plt.figure(1)
plt.plot(uu, B_sigmoid,'b--',label='Sigmoid')
plt.plot(uu,B,'r--',label='Standard' )
plt.legend()
plt.show()

plt.figure(2)
plt.plot(uu[1:], dB_sig,'b--',label='Sigmoid')
plt.plot(uu[1:],dB,'r--',label='Standard' )
plt.legend()
plt.show()
#plt.plot(uu[1:],dB)
#plt.plot(u, sigmoid(u))
# plt.plot(u,sigmoid(u-1), 'r--')
# plt.plot(u,-sigmoid(u-1), 'b--')
# plt.plot(u,sigmoid(u)-sigmoid(u-1), 'b-*')
# plt.plot(u,sigmoid_steep(u)-sigmoid_steep( u-1), 'r*')
# u_sign = -0.5 * (np.sign(-u) +1) + 0.5 * (np.sign(1-u) + 1)
# u_s = -0.5 * (sigmoid((np.sign(-u) +1)*1e5) )+ 0.5 * (sigmoid((np.sign(1-u) + 1) *1e5) )
# u_si = - (sigmoid(-u) +1) +  (sigmoid(1-u) + 1)
# u_ss = sigmoid(u_s)
# u_xx = (- (sigmoid((np.sign(-u) +1)*1e5) )+  (sigmoid((np.sign(1-u) + 1) *1e5) )) * 2
# print(u)
# print(u_s)
# print(u_si)
# print(u_ss)
# #plt.plot(u_s,'r')
# #plt.plot(u_si,'b')
# #plt.plot(sigmoid(-u))
# #plt.plot(sigmoid(u))
# #plt.plot(sigmoid(-u) + sigmoid(u))
#
#F_refine(uu)
# plt.plot(u_s,'b--')
# plt.plot(u_sign,'r--')
# plt.plot(u_tg,'b--')
# plt.plot(u_xx,'r*')

F   = lambda a,b,z : (H(z-a)-H(z-b)) * (a*ff((z-b)/(a-b)) + b*(1-ff((z-b)/(a-b)))) + (1-(H(z-a)-H(z-b))) * z if b>a else (H(z-b)-H(z-a)) * (b*(1-ff((z-b)/(a-b))) + a*(ff((z-b)/(a-b)))) + (1-(H(z-b)-H(z-a))) * z # construct the fractional flow curve
H   = lambda x : 1/(1+np.exp(-2*1e4*x)) # heaveside step function
ff  = lambda z : z**2 / ((z**2 + 5*(1-z)**2)) # fractional flow of gas
def conv(Fini,Finj,z,xini,yini):
    """
    Fini = fractional flow curve for the initial tie-line
    Finj = fractional flow curve for the injection tie-line
    z = pseudo-composition i.e. z_obl
    xini = end-point initial tie-line, for z<xini --> liquid
    yini = end-point initial tie-line, for z>yini --> vapour
    """
    overlap = (np.abs(Fini-z) > 1e-1) & (np.abs(Finj-z) > 1e-1)
    if len(Fini[overlap])!=0:
        index = np.where(np.abs(Fini-Finj) == np.min(np.abs(Fini[overlap]-Finj[overlap])))
        z_intersect=z[index]
        Fr = H(xini/yini-1)*(H(z-z_intersect)*Fini + (1-H(z-z_intersect))*Finj) + (1-H(xini/yini-1))*(H(z-z_intersect)*Finj + (1-H(z-z_intersect))*Fini)
    else:
        Fr = H(xini/yini-1)*Fini + (1-H(xini/yini-1))*Finj
    return Fr