"""
Created on November 3rd 10:02:16 2021
@author: amnedic
"""

import numpy as np
import scipy.linalg as la
import time
from multiprocessing import Pool
import numba as nb
from numba import njit
import gc

omega = 0.10 #finite frequency in eV
fil = 12.78 #filling of electrons per unit cell
mu = 6.1845 #chemical potential
kbT = 0.03 #temperature in eV
N = 25 #number of k-points for discretization of BZ: NxNxN
eta = 0.006
 
q_biglist_val = []

#GX-direction
n0 = 21 #asking 20 cores
q_val = np.arange(0, np.pi+10**(-15), np.pi/(n0-1))
[q_biglist_val.append([t, [[-q_val[t]/2, q_val[t]/2, 0]]]) for t in range(n0-1)]
q_num = len(q_biglist_val) #number of q-points

start_time = time.time()
#importing the Hamiltonian
latHam_data = np.genfromtxt('Wannier90/SrCo2As2/ham_wannier90_hr-SrCo2As2.dat')
L = len(latHam_data)
list_r = [[latHam_data[[i]][0][0], latHam_data[[i]][0][1], latHam_data[[i]][0][2]] for i in range(L)] #components of the position vectors in terms of the lattice vectors
list_nm = [[int(latHam_data[[i]][0][3]-1), int(latHam_data[[i]][0][4])-1] for i in range(L)] #orbital indices n and m (notation of orbitals 0-15)
list_re_im = [latHam_data[[i]][0][5]+1j*latHam_data[[i]][0][6] for i in range(L)] #hopping parameters in real space from Wannier Hamiltonian

weights_data = np.genfromtxt('Wannier90/SrCo2As2/nrpts_wannier90_hr-SrCo2As2.dat', delimiter='\n', dtype=str) #importing weights from Wannier90 output
weights_data_str = [str(t).split() for t in weights_data] 
c_weights_list = [1/int(item) for sublist in weights_data_str for item in sublist]  #c_weights_list is a flat list containing 1/weights
w = len(c_weights_list)

wfcenters_data = np.genfromtxt('Wannier90/SrCo2As2/wfcenters_wannier90_hr-SrCo2As2.dat') #centers of Wannier functions from Wannier90 output 
orb = len(wfcenters_data)
wann_R = [[wfcenters_data[[i]][0][0], wfcenters_data[[i]][0][1], wfcenters_data[[i]][0][2]] for i in range(orb)]
rotmat = np.genfromtxt('Wannier90/SrCo2As2/rotmat.dat') #C4 rotation of orbitals; the order of orbitals dz^2, dxz, dyz, dx^2-y^2, dxy, dz^2, dxz, dyz, dx^2-y^2, dxy, pz, px, py, pz, px, py

list_r_ar=np.array(list_r)
list_nm_ar=np.array(list_nm)
wann_R_ar=np.array(wann_R)
c_weights_list_ar=np.array(c_weights_list)
list_re_im_ar=np.array(list_re_im)

def convert_primitive_to_conventional(k_primitive): #converting points in momentum space from primitive [0, 2pi] to conventional coordinates [0, 1]
    k_conventional = []
    for k_val in k_primitive:
        k_xyz=[(k_val[1]+k_val[2])/(2*np.pi), (k_val[0]+k_val[2])/(2*np.pi), (k_val[0]+k_val[1])/(2*np.pi)]  
        k_conventional.append(k_xyz)   
    return k_conventional

#sampling k-biglist from 1BZ in primitive coordinates
k_val = np.arange(-np.pi, np.pi, 2*np.pi/N)
k_primitive_range = [[k_val[a]+1.37*10**(-12), k_val[b]-1.29*10**(-12), k_val[c]-1.13*10**(-12)] for a in range(len(k_val)) for b in range(len(k_val)) for c in range(len(k_val))]
p_xyzlist = convert_primitive_to_conventional(k_primitive_range) #converting points  to conventional coordinates
#sampling centers of neighboring cells in primitive coordinates
nn=2
nn_list = [[2*np.pi*a,2*np.pi*b,2*np.pi*c] for a in range(-nn,nn+1) for b in range(-nn,nn+1) for c in range(-nn,nn+1)]
nn_xyz_list = convert_primitive_to_conventional(nn_list)  #neighboring cells in conventional coordinates

def shifttoWScell(pq_xyz): #shift to central WS unit cell including all C4-symmetric partners
    norm_list = []
    for nn_xyz in nn_xyz_list: #going over the neigboring unit cells
        norm = np.linalg.norm([pq_xyz[0]-nn_xyz[0], pq_xyz[1]-nn_xyz[1], pq_xyz[2]-nn_xyz[2]]) #find in which unit cell this point lives
        norm_list.append(norm)
    ind = np.argmin(norm_list) #index of the nearest WS cell center
    norm_0 = np.linalg.norm([pq_xyz[0], pq_xyz[1], pq_xyz[2]]) #the distance to the central unit cell
    if norm_0<=norm_list[ind]: #if the point is already in the central WS cell
        pq_xyz_shift = pq_xyz
    else: #otherwise shift
        pq_xyz_shift = [pq_xyz[0]-nn_xyz_list[ind][0], pq_xyz[1]-nn_xyz_list[ind][1], pq_xyz[2]-nn_xyz_list[ind][2]] #shift the point to WS cell - note that these boundaries are still not treated correctly
    k_rot = [pq_xyz_shift[1], -pq_xyz_shift[0], pq_xyz_shift[2]] # -C4 rotation in conventional coord
    k_rot2 = [k_rot[1], -k_rot[0], k_rot[2]] # -C4^2 rotation in conventional coord
    k_rot3 = [k_rot2[1], -k_rot2[0], k_rot2[2]] # -C4^2 rotation in conventional coord    
    return (pq_xyz_shift, k_rot, k_rot2, k_rot3)

p_xyz_WS = []
for p_xyz in p_xyzlist: #adding all C4-symmetric points if they are not already in the list
    k_rot = [p_xyz[1], -p_xyz[0], p_xyz[2]] # -C4 rotation in conventional coord
    k_rot2 =  [k_rot[1], -k_rot[0], k_rot[2]] # -C4^2 rotation in conventional coord
    k_rot3 =  [k_rot2[1], -k_rot2[0], k_rot2[2]] # -C4^2 rotation in conventional coord
    if k_rot not in p_xyzlist:
        p_xyzlist.append(k_rot) #enforcing adding all C4-symmetric points if they are not already in the list
    if k_rot2 not in p_xyzlist:
        p_xyzlist.append(k_rot2)
    if k_rot3 not in p_xyzlist:
        p_xyzlist.append(k_rot3)

for p_xyz in p_xyzlist:
    norm_list = []
    for nn_xyz in nn_xyz_list: #going over the neigboring unit cells
        norm = np.linalg.norm([p_xyz[0]-nn_xyz[0], p_xyz[1]-nn_xyz[1], p_xyz[2]-nn_xyz[2]]) #find in which unit cell this point lives
        norm_list.append(norm)
    ind = np.argmin(norm_list) #index of the nearest unit cell
    ind2 = np.argpartition(norm_list, 1)[1] #second nearest unit cell
    norm_0 = np.linalg.norm([p_xyz[0], p_xyz[1], p_xyz[2]]) #the distance to the central unit cell
    if norm_0==norm_list[ind]: #if the point is already in the central WS cell
        p_xyz_shift = p_xyz
    else: #otherwise shift
        p_xyz_shift = [p_xyz[0]-nn_xyz_list[ind][0], p_xyz[1]-nn_xyz_list[ind][1], p_xyz[2]-nn_xyz_list[ind][2]] #shift the point to WS cell - note that these boundaries are still not treated correctly
    p_xyz_WS.append(p_xyz_shift)

print(np.shape(p_xyzlist), np.shape(p_xyz_WS))
k_biglist = []
for p_xyz_shift in p_xyz_WS: #convert from conventional to primitive coordinates
    k_biglist.append([np.pi*(p_xyz_shift[1]+p_xyz_shift[2]-p_xyz_shift[0]), np.pi*(p_xyz_shift[0]+p_xyz_shift[2]-p_xyz_shift[1]), np.pi*(p_xyz_shift[0]+p_xyz_shift[1]-p_xyz_shift[2])]) 
print(np.shape(k_biglist))

qvaluestr = str(q_biglist_val[0][0]) #gives the index of q-vector
q_biglist = q_biglist_val[0][1] #gives the q-vector
q_biglist_0 = [[0,0,0]] #we separatelly calculate for G=(0,0,0) point
    
#making a list of k+q values in format (q_biglist)(k_biglist)
kq_biglist = [[[q_biglist[q][0]+k_biglist[k][0], q_biglist[q][1]+k_biglist[k][1], q_biglist[q][2]+k_biglist[k][2]] for k in range(len(k_biglist))] for q in range(len(q_biglist))]
#and the same thing for list q_biglist_0 (to have the same shape as kq_biglist for running it in the same way)
kq_biglist_0 = [k_biglist for q in range(1)]    

@njit
def Ham_calc_nb(w,lk,m_biglist,k_prim,k_prim2,k_prim3,list_r,orb,list_nm,wann_R,c_weights_list,list_re_im):
    Ham = np.zeros((orb, orb, lk), dtype=nb.c16)
    Ham_rot = np.zeros((orb, orb, lk), dtype=nb.c16)
    Ham_rot2 = np.zeros((orb, orb, lk), dtype=nb.c16)
    Ham_rot3 = np.zeros((orb, orb, lk), dtype=nb.c16)
    for i in range(w):
    #defining the exponent with dot product x*kx + y*ky + z*kz which we will use for FT
    #it works faster if we include 1j prefactor already here
        list_r_orb=list_r[orb*orb*i]
        c_exp = np.exp(1j*((m_biglist*list_r_orb).sum(1)))
        c_exprot = np.exp(1j*((k_prim*list_r_orb).sum(1)))
        c_exprot2 = np.exp(1j*((k_prim2*list_r_orb).sum(1)))
        c_exprot3 = np.exp(1j*((k_prim3*list_r_orb).sum(1)))
        for o in range(orb**2):
            row = (orb**2)*i+o
            n = list_nm[row][0]
            m = list_nm[row][1]
            R_nm = np.array([(wann_R[n][0]-wann_R[m][0]), (wann_R[n][1]-wann_R[m][1]), (wann_R[n][2]-wann_R[m][2])])

            #calculating Hamiltonian in momentum space
            c= c_weights_list[i]*list_re_im[row]
            Ham[n][m] += c*c_exp*np.exp(-1j*((m_biglist*R_nm).sum(1)))
            Ham_rot[n][m] += c*c_exprot*np.exp(-1j*((k_prim*R_nm).sum(1))) #we calculate H(-C4 k)
            Ham_rot2[n][m] += c*c_exprot2*np.exp(-1j*((k_prim2*R_nm).sum(1))) #H(-C4^2 k)
            Ham_rot3[n][m] += c*c_exprot3*np.exp(-1j*((k_prim3*R_nm).sum(1))) #H(-C4^3 k)
    return Ham, Ham_rot, Ham_rot2, Ham_rot3

#parallel running
def calculate(q_biglist_val):
    qvaluestr = str(q_biglist_val[0]) #gives the index of q-vector
    q_biglist = q_biglist_val[1] #gives the q-vector
    q_biglist_0 = [[0,0,0]] #we separatelly calculate for G=(0,0,0) point
    
    #making a list of k+q values in format (q_biglist)(k_biglist)
    kq_biglist = [[[q_biglist[q][0]+k_biglist[k][0], q_biglist[q][1]+k_biglist[k][1], q_biglist[q][2]+k_biglist[k][2]] for k in range(len(k_biglist))] for q in range(len(q_biglist))]
    #and the same thing for list q_biglist_0 (to have the same shape as kq_biglist for running it in the same way)
    kq_biglist_0 = [k_biglist for q in range(1)]    

    epsilonnorm = 10**(-12)
    #calculating energies and eigenvectors - this function in general depends only on kq_biglist function (all information are here already), but for parallel running on cluster, it is defined also as a function of q_biglist    
    def energy(q_biglist,kq_biglist):
        all_eiglist = []
        all_eigvectors = []

        for q in range(0, len(q_biglist)):
            kq_list=kq_biglist[q]
            pq_xyz_list = [] #making a list in (conventional) coordinates
            for kq in kq_list:
                pq_xyz_list.append([(kq[1]+kq[2])/(2*np.pi), (kq[0]+kq[2])/(2*np.pi), (kq[0]+kq[1])/(2*np.pi)])   #in conventional coordinates 
        
            m_biglist = []
            k_prim = []
            k_prim2 = []
            k_prim3 = []
            for pq_xyz in pq_xyz_list:
                pq_xyz_shift = shifttoWScell(pq_xyz)[0] #function that shifts points to the central WS unit cell
                k_rot = shifttoWScell(pq_xyz)[1] # -C4 rotation in conventional coord
                k_rot2 = shifttoWScell(pq_xyz)[2] # -C4^2 rotation in conventional coord
                k_rot3 = shifttoWScell(pq_xyz)[3] # -C4^2 rotation in conventional coord
                    
                m_biglist.append([np.pi*(pq_xyz_shift[1]+pq_xyz_shift[2]-pq_xyz_shift[0]), np.pi*(pq_xyz_shift[0]+pq_xyz_shift[2]-pq_xyz_shift[1]), np.pi*(pq_xyz_shift[0]+pq_xyz_shift[1]-pq_xyz_shift[2])])
                k_prim.append([np.pi*(k_rot[1]+k_rot[2]-k_rot[0]), np.pi*(k_rot[0]+k_rot[2]-k_rot[1]), np.pi*(k_rot[0]+k_rot[1]-k_rot[2])]) 
                k_prim2.append([np.pi*(k_rot2[1]+k_rot2[2]-k_rot2[0]), np.pi*(k_rot2[0]+k_rot2[2]-k_rot2[1]), np.pi*(k_rot2[0]+k_rot2[1]-k_rot2[2])]) 
                k_prim3.append([np.pi*(k_rot3[1]+k_rot3[2]-k_rot3[0]), np.pi*(k_rot3[0]+k_rot3[2]-k_rot3[1]), np.pi*(k_rot3[0]+k_rot3[1]-k_rot3[2])]) 

            print(qvaluestr, np.shape(pq_xyz_list), np.shape(k_biglist), np.shape(m_biglist), np.shape(k_prim), np.shape(k_prim2), np.shape(k_prim3))

            eiglist = []
            eigvectors = []
            m_biglist_ar=np.array(m_biglist)
            k_prim_ar=np.array(k_prim)
            k_prim2_ar=np.array(k_prim2)
            k_prim3_ar=np.array(k_prim3)

            lk=len(k_biglist)
            Ham,Ham_rot,Ham_rot2,Ham_rot3=Ham_calc_nb(w,lk,m_biglist_ar,k_prim_ar,k_prim2_ar,k_prim3_ar,list_r_ar,orb,list_nm_ar,wann_R_ar,c_weights_list_ar,list_re_im_ar)

            # for i in range(w):
            #     #defining the exponent with dot product x*kx + y*ky + z*kz which we will use for FT
            #     #it works faster if we include 1j prefactor already here
            #     c_exp = np.exp(1j*(np.multiply(m_biglist, list_r[orb*orb*i]).sum(1)))
            #     c_exprot = np.exp(1j*(np.multiply(k_prim, list_r[orb*orb*i]).sum(1)))
            #     c_exprot2 = np.exp(1j*(np.multiply(k_prim2, list_r[orb*orb*i]).sum(1)))
            #     c_exprot3 = np.exp(1j*(np.multiply(k_prim3, list_r[orb*orb*i]).sum(1)))
            #     for o in range(orb**2):
            #         row = (orb**2)*i+o
            #         n = list_nm[row][0]
            #         m = list_nm[row][1]
            #         R_nm = [(wann_R[n][0]-wann_R[m][0]), (wann_R[n][1]-wann_R[m][1]), (wann_R[n][2]-wann_R[m][2])]
        
            #         #calculating Hamiltonian in momentum space
            #         c= c_weights_list[i]*list_re_im[row]
            #         Ham[n][m] += c*c_exp*np.exp(-1j*(np.multiply(m_biglist, R_nm).sum(1)))
            #         Ham_rot[n][m] += c*c_exprot*np.exp(-1j*(np.multiply(k_prim, R_nm).sum(1))) #we calculate H(-C4 k)
            #         Ham_rot2[n][m] += c*c_exprot2*np.exp(-1j*(np.multiply(k_prim2, R_nm).sum(1))) #H(-C4^2 k)
            #         Ham_rot3[n][m] += c*c_exprot3*np.exp(-1j*(np.multiply(k_prim3, R_nm).sum(1))) #H(-C4^3 k)

            #diagonalizing Hamiltonian for each k-value from our list separately                    
            for p in range(len(k_biglist)):
                rot0=Ham[:,:,p]
                rot1=np.matmul(rotmat,np.matmul(Ham_rot[:,:,p],np.linalg.inv(rotmat)))
                rot2=np.matmul(rotmat,np.matmul(np.matmul(rotmat,np.matmul(Ham_rot2[:,:,p],np.linalg.inv(rotmat))),np.linalg.inv(rotmat)))
                rot3=np.matmul(rotmat,np.matmul(np.matmul(rotmat,np.matmul(np.matmul(rotmat,np.matmul(Ham_rot3[:,:,p],np.linalg.inv(rotmat))),np.linalg.inv(rotmat))),np.linalg.inv(rotmat)))

                #for original Hamiltonian
                #Ham0 = rot0
                #eig, eigv = la.eigh(Ham0)        

                #for symmetrized Hamiltonian
                SymHam = 0.25*(rot0+rot1+rot2+rot3)
                eig, eigv = la.eigh(SymHam)

                eiglist.append(np.real(eig))
                eigvectors.append(eigv)

            all_eiglist.append(eiglist)
            all_eigvectors.append(eigvectors)
            
        return (all_eiglist, all_eigvectors)

    start_time = time.time()

    energylist = energy(q_biglist,kq_biglist)
    gc.collect()
    energylist1 = energy(q_biglist_0, kq_biglist_0)
    gc.collect()
    
    #eigenenergies and eigenvectors: ener & ener_v
    ener = energylist[0] #shape (q_biglist)(k_biglist)(orb)
    ener_v = energylist[1] #shape (q_biglist)(k_biglist)(orb)(orb)
    #np.save('fullA_test_ener_N='+str(N)+'_q='+str(q_biglist_val[0])+'.npy', ener)
    #np.save('fullA_test_ener_v_N='+str(N)+'_q='+str(q_biglist_val[0])+'.npy', ener_v)

    #eigenenergies and eigenvectors for q=0
    ener0 = energylist1[0][0] #shape (k_biglist)(orb)
    ener_v0 = energylist1[1][0] #shape (k_biglist)(orb)(orb)   

    del energylist
    del energylist1
    gc.collect()
    #to access eigenvectors as (bands)(components), one has to transpose them, as the python output for eigenvectors is (components)(bands)
    ener_v0_transp = [np.transpose(vec) for vec in ener_v0] #for q=0 the shape is (k_biglist, orb, orb)
    ener_v_transp = [[np.transpose(vec) for vec in vec_q] for vec_q in ener_v] #for all q: (q_biglist, k_biglist, orb, orb)

    #flat list of energies for q=0: (k_biglist, orb) to (k_biglist*orb)
    ener0_k_m = [item_k_n for ener0_k in ener0 for item_k_n in ener0_k]
    #flat list of energies for other q values: (q_biglist, k_biglist, orb) to (q_biglist*k_biglist*orb)
    ener_q_k_n = [item_q_k_n for ener_q in ener for ener_q_k in ener_q for item_q_k_n in ener_q_k]

    #fermi function - list of length (k_biglist*orb)
    nf0_k_m = [1/(np.exp((eig0_k_m-mu)/kbT)+1) for eig0_k_m in ener0_k_m]
    nf_q_k_n = [1/(np.exp((eig_q_k_n-mu)/kbT)+1) for eig_q_k_n in ener_q_k_n]
    
    del ener
    del ener_v
    del ener0
    del ener_v0
    gc.collect()
    #calculating the expression for susceptibility: (nf0_km - nf_qkn)/(eig0_km-eig_qkn)
    fermi_f_div_list = []
    for q in range(len(q_biglist)):
        for k in range(len(k_biglist)):
            for n in range(orb):
                index_qkn = q*len(k_biglist)*orb + k*orb + n
                eig_qkn = ener_q_k_n[index_qkn]
                nf_qkn = nf_q_k_n[index_qkn]
                for m in range(orb):
                    index0_km = k*orb+m            
                    eig0_km = ener0_k_m[index0_km]
                    nf0_km = nf0_k_m[index0_km]                
                    fermi_f_div = (nf0_km - nf_qkn)/(eig0_km-eig_qkn+omega+1j*eta)
                    fermi_f_div_list.append(fermi_f_div)    
    fermi_f_div_list_in=np.array(fermi_f_div_list)
    del fermi_f_div_list   
    gc.collect()    
    #for q=0, we will need the product (u_b^n(k))^*u_d^n(k): list(k_biglist*orb*orb*orb)
    ener_v0_k_m_c_d_in = np.array([item_c*np.conj(item_d) for ener_v0_k in ener_v0_transp for ener_v0_k_n in ener_v0_k for item_c in ener_v0_k_n for item_d in ener_v0_k_n])
    #for q\neq0, we will need the product (u_a^n(k+q))^*u_c^n(k+q): list(q_biglist*k_biglist*orb*orb*orb)
    ener_v_q_k_n_a_b_in = np.array([item_a*np.conj(item_b) for ener_v_q in ener_v_transp for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n for item_b in ener_v_q_k_n])                        

    del ener_v0_transp
    del ener_v_transp
    del ener0_k_m
    del ener_q_k_n
    del nf0_k_m
    del nf_q_k_n
    gc.collect()
    #here we do the summmation over k, m and n and calculate susceptibility as function of q and (a, b, c, d)
    k_biglist_in=np.array(k_biglist)

    time2 = round((time.time() - start_time), 2)
    print(time2)

    @njit
    def susceptibility_flat_n(a, b, c, d, k_biglist1, ener_v_q_k_n_a_b1, ener_v0_k_m_c_d1, fermi_f_div_list1, orb1):
        susc = 0+0.j
        q=0
        lk=len(k_biglist1)
        for k in range(lk):
            for n in range(orb1):
                una_unb = ener_v_q_k_n_a_b1[q*orb1*orb1*orb1*lk + k*orb1*orb1*orb1 + n*orb1*orb1 + a*orb1 + b]
                for m in range(orb1):
                    umc_umd = ener_v0_k_m_c_d1[k*orb1*orb1*orb1 + m*orb1*orb1 + c*orb1 + d]
                    fermi_f_div = fermi_f_div_list1[q*orb1*orb1*lk + k*orb1*orb1 + n*orb1 + m]
                    susc+=una_unb*umc_umd*fermi_f_div
        
        return np.array([-susc/lk])

    # def susceptibility_flat(a, b, c, d):
    #     sumlist_q = []
    #     for q in range(len(q_biglist)):
    #         susc = []
    #         for k in range(len(k_biglist)):
    #             for n in range(orb):
    #                 una_unb = ener_v_q_k_n_a_b[q*orb*orb*orb*len(k_biglist) + k*orb*orb*orb + n*orb*orb + a*orb + b]
    #                 for m in range(orb):
    #                     umc_umd = ener_v0_k_m_c_d[k*orb*orb*orb + m*orb*orb + c*orb + d]
    #                     fermi_f_div = fermi_f_div_list[q*orb*orb*len(k_biglist) + k*orb*orb + n*orb + m]
    #                     susc.append(una_unb*umc_umd*fermi_f_div)

    #         sumlist_q.append(np.sum(susc))
    #     return -np.array(sumlist_q)/(len(k_biglist))
    
    #all abcd
    start_time = time.time()
    susc = []
    print('time needed for calculating all channels of susceptibility for', len(q_biglist), 'q-values:')
    for a in range(orb):
        for d in range(orb):
            for b in range(orb):
                for c in range(orb):
                    #susc_adbc = susceptibility_flat(a, b, c, d)
                    susc_adbc= susceptibility_flat_n(a, b, c, d, k_biglist_in, ener_v_q_k_n_a_b_in, ener_v0_k_m_c_d_in, fermi_f_div_list_in, orb)
                    susc.append(susc_adbc)

#                    if a == d and b == c: #uncomment this block for calculating physical channel abba only
#                        #susc_adbc = susceptibility_flat(a, b, c, d)
#                        susc_adbc= susceptibility_flat_n(a, b, c, d, k_biglist_in, ener_v_q_k_n_a_b_in, ener_v0_k_m_c_d_in, fermi_f_div_list_in, orb)
#                        susc.append(susc_adbc)

                    if a == 0 and d == 0 and b == 0 and c == orb-1: #time estimation for full calculation
                        time2 = round((time.time() - start_time), 2)
                        print('estimated time needed:', round(time2*orb*orb*orb/60/60, 2), 'hrs')

        print("(a, d, b, c) = (", a, d, b, c, "), time =", time2/3600, "hrs and progres", round(a*100/orb, 2), "%")                 
    time2 = round((time.time() - start_time), 2)
    print(time2)
    print('DONE')

    pathW='/work/LAS/porth-lab/amnedic/'
    np.savetxt(pathW+'GX_N='+str(len(k_val))+'_fil='+str(fil)+'_omega='+str(omega)+'_q='+str(q_biglist_val[0])+'.dat', susc)
    return 1

if __name__ == '__main__':
    p = Pool(q_num)
    print(p.map(calculate, q_biglist_val))