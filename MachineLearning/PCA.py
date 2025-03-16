import numpy as np

Matrix_1 = np.array([[1,2,3],[2,4,6],[3,4,5],[6,7,8],[7,8,9]])
[Row,Col] = Matrix_1.shape
print('Row:\n',Row)
print('Column:\n',Col)
print(Matrix_1)

'''
Step.1 
Mean vector calculate
'''
Column_1_mean = np.mean(Matrix_1[:,0])
Column_2_mean = np.mean(Matrix_1[:,1])
Column_3_mean = np.mean(Matrix_1[:,2])
print('C1 = {},C2= {}, C3= {}'.format(Column_1_mean,Column_2_mean,Column_3_mean))

mean_vector = np.array([[Column_1_mean,Column_2_mean,Column_3_mean]])
print('Mean vector :\n',mean_vector)

'''
Step.2
Covariance matrix calculate
'''
for i in range(0,Row+1):
    Matrix_mean = Matrix_1[:i] - mean_vector
Cov_Matrix_Mean = np.cov([Matrix_mean[:,0],Matrix_mean[:,1],Matrix_mean[:,2]])
print('Mean Matrix:\n',Matrix_mean)
print('Covariance matrix: \n',Cov_Matrix_Mean)

'''
Step.3
EigenValues and EigenVectors calculate
'''
eigen_Value,eigen_Vctor = np.linalg.eig(Cov_Matrix_Mean)
print(eigen_Vctor)
for i in range(len(eigen_Value)):
    print('Eigen Value{} = {}\nEigen vector{} = \n{}\n'.format(i+1,eigen_Value[i],i+1,eigen_Vctor[:,i].reshape(1,3).T))

'''
Step.4
Check EigenValues and EigenVectors
'''

for i in range(len(eigen_Value)):
    a = np.array(np.dot(Cov_Matrix_Mean, eigen_Vctor[:, i]))
    b = np.array(eigen_Value[i] * eigen_Vctor[:, i])
    print('left:',np.dot(Cov_Matrix_Mean,eigen_Vctor[:,i]))
    print('right:',eigen_Value[i]*eigen_Vctor[:,i])
    np.testing.assert_array_almost_equal(a,b,decimal=6, err_msg='', verbose=True)

'''
Step.5
Sort eigenvectors and values
'''
for j in eigen_Vctor:
    np.testing.assert_array_almost_equal(1.0,np.linalg.norm(j))

#store eigenvalue and eigenvector in (|x|,array[]) foramt
SetComparison = [(np.abs(eigen_Value[i]), eigen_Vctor[:,i]) for i in range(len(eigen_Value))]

#Sort the (eigenvalue, eigenvector) tuples from high to low
SetComparison.sort(key=lambda x: x[0], reverse=True)
for i in SetComparison:
    print('EigenVector after sorted = {}\n'.format(i[0]))

'''Step.6 Choosing K eigenvectors with the largest eigenvalues'''
Matrix_EigenVector = np.hstack((SetComparison[0][1].reshape(3,1), SetComparison[1][1].reshape(3,1)))
print('Eigenvector matrix after dimensionality reduction:\n{}\n'.format(Matrix_EigenVector))


'''step.7 Transforming the samples onto the new subspace'''
Transformed_Matrix = Matrix_1.dot(Matrix_EigenVector)

print('The dimensionality reduction matrix is:\n{}\n'.format(Transformed_Matrix))

