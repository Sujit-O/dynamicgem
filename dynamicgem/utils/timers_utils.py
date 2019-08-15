import os
import numpy as np
from scipy.sparse import csc_matrix, find
from scipy.sparse.linalg import svds, eigs
from numpy import linalg as LA


output = './output'
def hasmapping(filepath):
	"""Function to map arbitrary node numbers to consecutive integer values.
           
        Args:
            datapath (str): Txt path where the graph is saved.

        Returns:
        	Dictionary: Dictionary of arbitrary node to its consecutive integer values mapping.
    """
	with open(filepath) as fp:
		line = fp.readline()
		line = fp.readline()
		line = fp.readline()
		# first three lines are metadata store by networkx
		line = fp.readline()
		cnt = 0
		M = {}
		while line:
			line = line.split('\n')[0]
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
		# import pdb 
		# pdb.set_trace()
		line = fp.readline()
		line = fp.readline()
		line = fp.readline()
		# first three lines are metadata stored by networkx
		line = fp.readline()

		while line:
			line = line.split('\n')[0]
			node = line.split(' ')
			fromNode = M[int(node[0])]
			l = len(node)
			if l>1:
				for i in range(1,l):
					toNode = M[int(node[i])]
					row.append(fromNode)
					col.append(toNode)
					data.append(1.0)
			line=fp.readline()

	return csc_matrix((data, (row, col)), shape=(len(M), len(M)), dtype= float)

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

def deltaA(A, filepath, M):
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
	S_delta = csc_matrix(([0], ([0], [0])), shape=(L,L), dtype = float)

	[i_old, j_old, val_old] = find(A)
	[i_new, j_new, val_new] = find(A_new)

	for k in range(len(i_new)):
		temp_old = A[i_new[k],j_new[k]]
		S_delta[i_new[k],j_new[k]]=val_new[k]-temp_old

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

    # solve eigenvalue and eigenvectors from SVD, denote as L, X
	Old_X = Old_U
	for i in range(K):
		temp_i = np.argmax(np.absolute(Old_X[:,i]))
		if Old_X[temp_i,i]<0:
			Old_X[:,i]= - Old_X[:,i]
	temp_v = Old_U.max(axis=0)
	temp_i = np.argmax(Old_U, axis = 0)
	# import pdb
	# pdb.set_trace()
	ind = [np.ravel_multi_index((i, p), dims=(N,K), order='F') for i,p in zip(temp_i,range(K))]
	temp_sign = np.sign(temp_v*[Old_V.ravel()[j] for j in ind])
	Old_L = np.multiply(np.diag(Old_S),temp_sign)

    # calculate the sum term
	temp_sum = np.transpose(Old_X)@ Delta_A.toarray()@ Old_X
    #calculate eignevalues of changes
	Delta_L = np.transpose(np.diag(temp_sum))

    #calculate eigenvectors of change
	Delta_X = np.zeros([N,K])
	for i in range(K):
		temp_D = np.diag(np.ones([1,K])*(Old_L[i]-Delta_L[i])-Old_L)
		temp_alpha = LA.pinv(temp_D - temp_sum) @ temp_sum[:,i]
		Delta_X[:,i]= Old_X @ temp_alpha

    #return updated result
	New_U = Old_X + Delta_X
	for i in range(K):
		New_U[:,i]= New_U[:,i] / np.sqrt(np.transpose(New_U[:,i])@ New_U[:,i])
	New_S = np.diag(np.absolute(Old_L+Delta_L))
	New_V = New_U*np.diag(np.sign(Old_L+ Delta_L))
	# import pdb
	# pdb.set_trace()
	return New_U, New_S, New_V

def Obj_SimChange(S_ori, S_add, U, V):
	"""Function to  calculate the objective funciton or loss.
           
        Args:
            S_ori (Sparse Matrix): Sparse scipy of original adjancey matrix
            S_add (Sparse Matrix): Sparse scipy of added adjancey matrix
            U (Matrix): N x K left embedding vector
            V (Matrix): N x K right embedding vector
            loss_ori (float): Origina loss value
	
        Returns:
        	Float: New calculcated loss value
    """
	return LA.norm(S_ori.toarray()+S_add.toarray()- np.matmul(U,np.transpose(V)))

def getAddedEdge(A, filepath, M):
	"""Function to  get the difference between previous and current graph.
           
        Args:
            A (Sparse Matrix): Sparse scipy adjancey matrix
            M (dict): Dictionary of aribitrary nodes to consecutive node values
            filepath (str): Txt path where the graph is stored.
	
        Returns:
        	Sparse Matrix: The added graph
    """
	A_new = parseData(filepath, M)
	L = A.shape[0]
	S_add = csc_matrix(([0], ([0], [0])), shape=(L,L), dtype = float) 
	[i_new, j_new, val_new] = find(A_new)
	for k in range(len(i_new)):
		if A.toarray()[i_new[k],j_new[k]] != val_new[k]:
			S_add[i_new[k],j_new[k]] = val_new[k]

	return S_add

def RefineBound(S_ori, S_add, Loss_ori, K):
	"""Function to  calculate the objective funciton or loss.
       
       Loss_Bound = Loss_ori + trace_change(S x S^T) - eigs(delta(S x S^T),K)

        Args:
            S_ori (Sparse Matrix): Sparse scipy of original adjancey matrix
            S_add (Sparse Matrix): Sparse scipy of added adjancey matrix
            K (int): Embedding dimension
            loss_ori (float): Original loss value
	
        Returns:
        	Float: New calculcated loss bound
    """
    # Calculate the trace change
	
	S_overlap = (S_add != 0).multiply(S_ori)
	S_temp = S_add + S_overlap
	trace_change = csc_matrix.sum(S_temp.multiply(S_temp) - S_overlap.multiply(S_overlap))

	# Calculate eigenvalues sum of delta(S * S^T)
  	# Note: we only need to deal with non-zero rows/columns
	S_temp = S_ori.dot(S_add)

	# import pdb
	# pdb.set_trace()
	S_temp = S_temp + S_temp.transpose() + S_add.dot(S_add)
	# _,S_choose,_ = find(csc_matrix.sum(S_temp, axis=0))
	# S_temp = S_temp[S_choose,S_choose]

	temp_eigs,_ = eigs(S_temp, min(2*K, S_temp.shape[0]))
	temp_eigs = np.absolute(temp_eigs)
	temp_eigs =temp_eigs[temp_eigs>=0]
	temp_eigs=np.sort(temp_eigs)[::-1]

	if len(temp_eigs)>=K:
		eigen_sum = sum(temp_eigs[:K])
	else:
		temp_l = len(temp_eigs)
		eigen_sum = sum(temp_eigs)+temp_eigs[temp_l-1]*(K-temp_l)


	return Loss_ori + trace_change - eigen_sum

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

	#  Calculate Original Similarity Matrix
	# In this reference implementation, we assume similarity is 
	# adjacency matrix. Other variants shoule be straight-forward.
	print("Creating a hashmap for arbitrary node name to consecutive ones!")
	M = hasmapping(dataFolder+'/0')
	# Get the first time slice data for initialization
	print("Parsing the data for initialization!")
	A = parseData(dataFolder+'/0', M)
	# Initialize the symmetric matrix
	Sim = A
	N = A.shape[0]

	# calculate static solution
	print("Creating the svds for the first time_slice!")
	# import pdb
	# pdb.set_trace()
	U[0], S[0], V[0] = svds(A, k=K)

	U_cur = U[0] * np.sqrt(S[0])
	V_cur = np.transpose(V[0]) * np.sqrt(S[0])

	if not os.path.exists(output):
		os.mkdir(output)

	if not os.path.exists(output+'/'+datatype):
		os.mkdir(output+'/'+datatype)

	# save the current embeddings
	
	with open(output+'/'+datatype +'/0_U.txt','wb') as fh:
		for line in np.asmatrix(U_cur):
			np.savetxt(fh, line, fmt='%.4f')

	with open(output+'/'+datatype +'/0_V.txt','wb') as fh:
		for line in np.asmatrix(V_cur):
			np.savetxt(fh, line, fmt='%.4f')

	# get the current loss
	print("Calculating the loss for first time slice!")
	Loss_store[0] = Obj(Sim, U_cur, V_cur)
	print("Loss for first time slice:", Loss_store[0])
	# assign the bound as current loss
	Loss_bound[0] = Loss_store[0]

	# Store the cummulative similairty matrix
	S_cum = Sim
	# Store the cumulative perturbation from last rerun
	S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N), dtype=float)

	loss_rerun = Loss_store[0]

	for i in range(1,time_slice):
		print("calculating the embedding for time slice:",i)
		S_add = deltaA(S_cum,dataFolder+'/'+str(i),M)
		S_perturb = S_perturb + S_add

		if (Update):
			U[i], S[i], V[i] = TRIP(U[i-1],S[i-1],V[i-1],S_add)
			# Note: TRIP does not insure smaller value

			U_cur = U[i] * np.sqrt(S[i])
			V_cur = V[i] * np.sqrt(S[i])

			# save the current embeddings		
			with open(output+'/'+datatype +'/incrementalSVD/'+str(i)+'_U.txt','wb') as fh:
				for line in U_cur:
					np.savetxt(fh, line, fmt='%.4f')

			with open(output+'/'+datatype +'/incrementalSVD/'+str(i)+'_V.txt','wb') as fh:
				for line in V_cur:
					np.savetxt(fh, line, fmt='%.4f')

			Loss_store[i] = Obj(S_cum+S_add, U_cur, V_cur)
		else:
			Loss_store[i] = Obj_SimChange(S_cum,S_add,U_cur,V_cur)

		Loss_bound[i] = RefineBound(Sim,S_perturb,loss_rerun,K)	
		S_cum = S_cum + S_add

		if Loss_store[i]>= (1+Theta) * Loss_bound[i]:
			print("Begin rerun at time stamp:", i)
			Sim = S_cum
			S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N),dtype=float)
			run_times = run_times +1
			Run_t[run_times] = i

			U[i], S[i], V[i] = svds(Sim, K)
			U_cur = U[i] * np.sqrt(S[i])
			V_cur = np.transpose(V[i]) * np.sqrt(S[i])

			loss_rerun = Obj(Sim,U_cur,V_cur);
			Loss_store[i] = loss_rerun
			Loss_bound[i] = loss_rerun

			# save the current embeddings		
			with open(output+'/'+datatype +'/rerunSVD/'+str(i)+'_U.txt','wb') as fh:
				for line in U_cur:
					np.savetxt(fh, line, fmt='%.4f')

			with open(output+'/'+datatype +'/rerunSVD/'+str(i)+'_V.txt','wb') as fh:
				for line in V_cur:
					np.savetxt(fh, line, fmt='%.4f')

	# Evaluation
	Loss_optimal = {}
	Loss_optimal[0] = Loss_store[0]
	Sim = A

	S_cum = Sim

	for i in range(1, time_slice):
		S_add = getAddedEdge(S_cum,dataFolder+'/'+str(i),M)
		S_cum = S_cum + S_add
		temp_U, temp_S, temp_V = svds(S_cum, k=K)

		# import pdb
		# pdb.set_trace()
		temp_U = temp_U * np.sqrt(temp_S)
		temp_V = np.transpose(temp_V) * np.sqrt(temp_S)

		# save the current embeddings		
		with open(output+'/'+datatype +'/optimalSVD/'+str(i)+'_U.txt','wb') as fh:
			for line in temp_U:
				np.savetxt(fh, line, fmt='%.4f')

		with open(output+'/'+datatype +'/optimalSVD/'+str(i)+'_V.txt','wb') as fh:
			for line in temp_V:
				np.savetxt(fh, line, fmt='%.4f')

		Loss_optimal[i] = Obj(S_cum, temp_U, temp_V)

		print("Optimal Loss for ", i, ":", Loss_optimal[i])



def incrementalSVD(dataFolder, K, Theta, datatype):
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

	#  Calculate Original Similarity Matrix
	# In this reference implementation, we assume similarity is 
	# adjacency matrix. Other variants shoule be straight-forward.
	print("Creating a hashmap for arbitrary node name to consecutive ones!")
	M = hasmapping(dataFolder+'/0')
	# Get the first time slice data for initialization
	print("Parsing the data for initialization!")
	A = parseData(dataFolder+'/0', M)
	# Initialize the symmetric matrix
	Sim = A
	N = A.shape[0]

	# calculate static solution
	print("Creating the svds for the first time_slice!")
	# import pdb
	# pdb.set_trace()
	U[0], S[0], V[0] = svds(A, k=K)

	U_cur = U[0] * np.sqrt(S[0])
	V_cur = np.transpose(V[0]) * np.sqrt(S[0])

	if not os.path.exists(output):
		os.mkdir(output)

	if not os.path.exists(output+'/'+datatype):
		os.mkdir(output+'/'+datatype)

	# save the current embeddings
	
	with open(output+'/'+datatype +'/0_U.txt','wb') as fh:
		for line in np.asmatrix(U_cur):
			np.savetxt(fh, line, fmt='%.4f')

	with open(output+'/'+datatype +'/0_V.txt','wb') as fh:
		for line in np.asmatrix(V_cur):
			np.savetxt(fh, line, fmt='%.4f')

	# get the current loss
	print("Calculating the loss for first time slice!")
	Loss_store[0] = Obj(Sim, U_cur, V_cur)
	print("Loss for first time slice:", Loss_store[0])
	# assign the bound as current loss
	Loss_bound[0] = Loss_store[0]

	# Store the cummulative similairty matrix
	S_cum = Sim
	# Store the cumulative perturbation from last rerun
	S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N), dtype=float)

	loss_rerun = Loss_store[0]

	for i in range(1,time_slice):
		print("calculating the embedding for time slice:",i)
		S_add = deltaA(S_cum,dataFolder+'/'+str(i),M)
		S_perturb = S_perturb + S_add

		if (Update):
			U[i], S[i], V[i] = TRIP(U[i-1],S[i-1],V[i-1],S_add)
			# Note: TRIP does not insure smaller value

			U_cur = U[i] * np.sqrt(S[i])
			V_cur = V[i] * np.sqrt(S[i])

			# save the current embeddings		
			with open(output+'/'+datatype +'/incrementalSVD/'+str(i)+'_U.txt','wb') as fh:
				for line in U_cur:
					np.savetxt(fh, line, fmt='%.4f')

			with open(output+'/'+datatype +'/incrementalSVD/'+str(i)+'_V.txt','wb') as fh:
				for line in V_cur:
					np.savetxt(fh, line, fmt='%.4f')


def optimalSVD(dataFolder, K, Theta, datatype):
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

	#  Calculate Original Similarity Matrix
	# In this reference implementation, we assume similarity is 
	# adjacency matrix. Other variants shoule be straight-forward.
	print("Creating a hashmap for arbitrary node name to consecutive ones!")
	M = hasmapping(dataFolder+'/0')
	# Get the first time slice data for initialization
	print("Parsing the data for initialization!")
	A = parseData(dataFolder+'/0', M)
	# Initialize the symmetric matrix
	Sim = A
	N = A.shape[0]

	# calculate static solution
	print("Creating the svds for the first time_slice!")
	# import pdb
	# pdb.set_trace()
	U[0], S[0], V[0] = svds(A, k=K)

	U_cur = U[0] * np.sqrt(S[0])
	V_cur = np.transpose(V[0]) * np.sqrt(S[0])

	if not os.path.exists(output):
		os.mkdir(output)

	if not os.path.exists(output+'/'+datatype):
		os.mkdir(output+'/'+datatype)

	# save the current embeddings
	
	with open(output+'/'+datatype +'/0_U.txt','wb') as fh:
		for line in np.asmatrix(U_cur):
			np.savetxt(fh, line, fmt='%.4f')

	with open(output+'/'+datatype +'/0_V.txt','wb') as fh:
		for line in np.asmatrix(V_cur):
			np.savetxt(fh, line, fmt='%.4f')

	# get the current loss
	print("Calculating the loss for first time slice!")
	Loss_store[0] = Obj(Sim, U_cur, V_cur)
	print("Loss for first time slice:", Loss_store[0])
	# assign the bound as current loss

	# Evaluation
	Loss_optimal = {}
	Loss_optimal[0] = Loss_store[0]
	Sim = A

	S_cum = Sim

	for i in range(1, time_slice):
		S_add = getAddedEdge(S_cum,dataFolder+'/'+str(i),M)
		S_cum = S_cum + S_add
		temp_U, temp_S, temp_V = svds(S_cum, k=K)

		# import pdb
		# pdb.set_trace()
		temp_U = temp_U * np.sqrt(temp_S)
		temp_V = np.transpose(temp_V) * np.sqrt(temp_S)

		# save the current embeddings		
		with open(output+'/'+datatype +'/optimalSVD/'+str(i)+'_U.txt','wb') as fh:
			for line in temp_U:
				np.savetxt(fh, line, fmt='%.4f')

		with open(output+'/'+datatype +'/optimalSVD/'+str(i)+'_V.txt','wb') as fh:
			for line in temp_V:
				np.savetxt(fh, line, fmt='%.4f')

		Loss_optimal[i] = Obj(S_cum, temp_U, temp_V)

		print("Optimal Loss for ", i, ":", Loss_optimal[i])


def rerunSVD(dataFolder, K, Theta, datatype):
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

	#  Calculate Original Similarity Matrix
	# In this reference implementation, we assume similarity is 
	# adjacency matrix. Other variants shoule be straight-forward.
	print("Creating a hashmap for arbitrary node name to consecutive ones!")
	M = hasmapping(dataFolder+'/0')
	# Get the first time slice data for initialization
	print("Parsing the data for initialization!")
	A = parseData(dataFolder+'/0', M)
	# Initialize the symmetric matrix
	Sim = A
	N = A.shape[0]

	# calculate static solution
	print("Creating the svds for the first time_slice!")
	# import pdb
	# pdb.set_trace()
	U[0], S[0], V[0] = svds(A, k=K)

	U_cur = U[0] * np.sqrt(S[0])
	V_cur = np.transpose(V[0]) * np.sqrt(S[0])

	if not os.path.exists(output):
		os.mkdir(output)

	if not os.path.exists(output+'/'+datatype):
		os.mkdir(output+'/'+datatype)

	# save the current embeddings
	
	with open(output+'/'+datatype +'/0_U.txt','wb') as fh:
		for line in np.asmatrix(U_cur):
			np.savetxt(fh, line, fmt='%.4f')

	with open(output+'/'+datatype +'/0_V.txt','wb') as fh:
		for line in np.asmatrix(V_cur):
			np.savetxt(fh, line, fmt='%.4f')

	# get the current loss
	print("Calculating the loss for first time slice!")
	Loss_store[0] = Obj(Sim, U_cur, V_cur)
	print("Loss for first time slice:", Loss_store[0])
	# assign the bound as current loss
	Loss_bound[0] = Loss_store[0]

	# Store the cummulative similairty matrix
	S_cum = Sim
	# Store the cumulative perturbation from last rerun
	S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N), dtype=float)

	loss_rerun = Loss_store[0]

	for i in range(1,time_slice):
		print("calculating the embedding for time slice:",i)
		S_add = deltaA(S_cum,dataFolder+'/'+str(i),M)
		S_perturb = S_perturb + S_add

		if (Update):
			U[i], S[i], V[i] = TRIP(U[i-1],S[i-1],V[i-1],S_add)
			# Note: TRIP does not insure smaller value

			U_cur = U[i] * np.sqrt(S[i])
			V_cur = V[i] * np.sqrt(S[i])

			Loss_store[i] = Obj(S_cum+S_add, U_cur, V_cur)
		else:
			Loss_store[i] = Obj_SimChange(S_cum,S_add,U_cur,V_cur)

		Loss_bound[i] = RefineBound(Sim,S_perturb,loss_rerun,K)	
		S_cum = S_cum + S_add

		if Loss_store[i]>= (1+Theta) * Loss_bound[i]:
			print("Begin rerun at time stamp:", i)
			Sim = S_cum
			S_perturb = csc_matrix(([0], ([0], [0])), shape=(N, N),dtype=float)
			run_times = run_times +1
			Run_t[run_times] = i

			U[i], S[i], V[i] = svds(Sim, K)
			U_cur = U[i] * np.sqrt(S[i])
			V_cur = np.transpose(V[i]) * np.sqrt(S[i])

			loss_rerun = Obj(Sim,U_cur,V_cur);
			Loss_store[i] = loss_rerun
			Loss_bound[i] = loss_rerun

			# save the current embeddings		
			with open(output+'/'+datatype +'/rerunSVD/'+str(i)+'_U.txt','wb') as fh:
				for line in U_cur:
					np.savetxt(fh, line, fmt='%.4f')

			with open(output+'/'+datatype +'/rerunSVD/'+str(i)+'_V.txt','wb') as fh:
				for line in V_cur:
					np.savetxt(fh, line, fmt='%.4f')
