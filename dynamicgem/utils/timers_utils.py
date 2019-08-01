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

	return csr_matrix((data, (row, col)), shape=(len(M), len(M)))

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

  	# calculate static solution
  	U[0], S[0], Vh[0] = svds(A, K)

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








