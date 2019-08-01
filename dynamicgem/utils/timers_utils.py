import os
import numpy as np
from scipy.sparse import csc_matrix, find
from scipy.sparse.linalg import svds
from numpy import linalg as LA


output = '../output'
def hasmapping(filepath):
	"""Function to map arbitrary node numbers to consecutive integer values.
           
        Args:
            datapath (str): Txt path where the graph is saved.

        Returns:
        	Dictionary: Dictionary of arbitrary node to its consecutive integer values mapping.
    """
	with open(filepath) as fp:
		line = fp.readline()
		cnt = 0
		M = {}
		while line:
			node = int(line.split(' ')[0])
			M[node] =cnt
			line = fp.readline()
			cnt += 1
	return M

def parseData(filepath, M):
	"""Function to read the custom data type of graph and return the adjacency matrix.
           
        Args:
            filepath (str): Txt path where the graph is stored.
            M (dict): Dictionary of aribitrary nodes to consecutive node values

        Returns:
        	Sparse: Sparse adjacency matrix.
    """
    # row, column and data to be passed to the sparse matrix
    row = []
    col = []
    data =[]
    with open(filepath) as fp:
		line = fp.readline()
		while line:
			node = line.split(' ')
			fromNode = M[int(node[0])]
			l = len(node)
			if l>1:
				for i in range(1,l):
					toNode = M[int(node[i])]
					row.append(fromNode)
					col.append(toNode)
					data.append(1)

	return csc_matrix((data, (row, col)), shape=(len(M), len(M)))

def Obj(Sim, U, V):
	"""Function to  calculate the objective funciton or loss.
           
        Args:
            Sim (Sparse Matrix): Sparse scipy adjancey matrix
            U (Matrix): N x K left embedding vector
            V (Matrix): N x K right embedding vector
	
        Returns:
        	Float: returns || S - U * V^T ||_F^2
    """

    # PS_u PS_v are the K x K matrix, pre-calculated sum for embedding vector
	# PS_u(i,j) = sum_k=1^N U(k,i)U(k,j)
	# PS_v(i,j) = sum_k=1^N V(k,i)V(k,j)

	return LA.norm(Sim.toarray()- np.matmul(U,np.transpose(V)))
    # PS_u = np.matmul(np.transpose(U) , U)
    # PS_v = np.matmul(np.transpose(V) , V)

    # [temp_row, temp_col, temp_value] = find(Sim)

    # # calculate the first term
    # L = np.sum(np.multiply(temp_value, temp_value))

    # # calculate the second term
    # M = len(temp_value)

    # # The calculation is separated to k iteration to avoid memory overflow
    # for i in range(K):
    # 	start_index = np.floor(i*M/K)
    # 	end_index = np.floor(i*M/K+1)
    # 	temp_inner = np.sum(np.multiply(U[temp_row[start_index:end_index],:],
    # 		V[temp_col[start_index:end_index],:]),axis=1)
    # 	L = L-2*np.sum(np.multiply(temp_value[start_index:end_index],temp_inner))
    # # calculate the third term
    # L = L+np.sum(np.multiply(PS_u,PS_v))

    # return L

def deltaA(A, filepath,M):
	"""Function to  get the difference between previous and current graph.
           
        Args:
            A (Sparse Matrix): Sparse scipy adjancey matrix
            M (dict): Dictionary of aribitrary nodes to consecutive node values
            filepath (str): Txt path where the graph is stored.
	
        Returns:
        	Sparse Matrix: The delta graph
    """
    A_new = parseData(filepath,M)
    L = A.shape[0]
    S_delta = csc_matrix(([0], ([0], [0])), shape=(L,L))

    [i_old, j_old, val_old] = find(A)
	[i_new, j_new, val_new] = find(A_new)

	for k in range(len(i_new)):
		temp_old = A[i_new[k],j_new[k]]
		S_delta[i_new(k),j_new(k)]=val_new[k]-temp_old

	for k in range(len(i_old)):
		temp_new = A_new[i_old[k],j_old[k]]
		S_delta[i_old[k],j_old[k]]=temp_new-val_old[k]

	return S_delta

def TRIP(Old_U,Old_S,Old_V, Delta_A):
	"""Function to  calculate the new embedding by utilizing the differene graph.

	 	Update the embedding using TRIP method reference: Chen Chen, and Hanghang Tong. 
		"Fast eigen-functions tracking on dynamic graphs." SDM, 2015.
           
        Args:
            Old_U (Matrix): old N x K left embedding vector
            Old_S (Matrix): old K x K  embedding vector
            Old_V (Matrix): old N x K right embedding vector
            DeltaA (Sparse Matrix): The difference of the two graphs.
	
        Returns:
        	 Matrices: New_U, New_S, New_V
    """
    N, K = np.shape(Old_U)
    Old_X = Old_U
    for i in range(K):
    	temp_i = np.argmax(np.absolute(Old_X[:,i]))
    	if Old_X[temp_i,i]<0:
    		Old_X[:,i]= - Old_X[:,i]
    temp_v = Old_U.max()
    temp_i = np.argmax(Old_U)

    



def TIMERS(dataFolder, K, Theta, datatype):
	"""Main timer function to perform embedding.
           
        Args:
            dataFolder (str): directory with the test data consiting of numbers to denote time varying dynamic graph.
            K (float): Embedding dimension
      		Theta (float): a threshold for re-run SVD
      		datatype (str): type of graph 

    """
    # Update the embedding
    Update = 1
    time_slice = len(os.listdir(dataFolder))
    # dictionary to store the USV values
    U = {}
    S = {}
    V = {}
    # store loss for each time stamp
    Loss_store = {}
    # store loss bound for each time stamp
    Loss_bound = {}
    # store how many time the rerun is executed
    run_times = 1
    # store the rerun based on time slice
    Run_t = np.zeros([time_slice+1,1])

    #  % Calculate Original Similarity Matrix
  	# In this reference implementation, we assume similarity is 
  	# adjacency matrix. Other variants shoule be straight-forward.
  	M = hasmapping(dataFolder+'/0')
  	# Get the first time slice data for initialization
  	A = parseData(dataFolder+'/0', M)
  	# Initialize the symmetric matrix
  	Sim = A
  	N = A.shape[0]

  	# calculate static solution
  	U[0], S[0], V[0] = svds(A, K)

  	U_cur = U[0] * np.sqrt(S[0])
  	V_cur - S[0] * np.sqrt(S[0])

  	if not os.path.exists(output):
  		os.mkdir(output)

  	# save the current embeddings		
  	with open(output+'/'+datatype +'/0_U.txt','wb') as fh:
	    for line in U_cur:
	        np.savetxt(fh, line, fmt='%.4f')

	with open(output+'/'+datatype +'/0_V.txt','wb') as fh:
	    for line in V_cur:
	        np.savetxt(fh, line, fmt='%.4f')

	# get the current loss
	Loss_store[0] = Obj(Sim, U_cur, V_cur)
	# assign the bound as current loss
	Loss_bound[0] = Loss_store[0]

	# Store the cummulative similairty matrix
	S_cum = Sim
	# Store the cumulative perturbation from last rerun
	S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N))

	loss_rerun = Loss_store[0]

	for i in range(1,time_slice):
		S_add = deltaA(S_cum,dataFolder+'/'+str(i),M)
		S_perturb = S_perturb + S_add

		if (Update):
			U[i], S[i], V[i] = TRIP(U[i-1],S[i-1],V[i-1],S_add)
			# Note: TRIP does not insure smaller value












