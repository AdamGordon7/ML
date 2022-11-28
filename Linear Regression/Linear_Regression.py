import numpy as np

def design_matrix(data):
    design_matrix=[]
    for i in data:
        row=[1,i]
        design_matrix.append(row)

    design_matrix=np.array(design_matrix)
    return design_matrix
    #print(design_matrix)

def closed_form_soln(training_set,y):
    training_design_matrix=design_matrix(training_set)
    theta=np.linalg.inv(training_design_matrix.transpose().dot(training_design_matrix)).dot(training_design_matrix.transpose()).dot(y)
    return theta

def gradient_decent(x,y,alpha,theta,num_its):
    theta0=theta[0]
    theta1=theta[1]
    for i in range(num_its):
        for j in range(len(x)):
            predicted_val=theta0+(theta1*x[j])
            loss=predicted_val-y[j]
            theta0=theta0-alpha*(loss)
            theta1=theta1-alpha*(loss)*x[j]
        
    theta=[theta0,theta1]
    return theta



x=[1,2,3,4,5]
y=[1,3,2,3,5]

theta=closed_form_soln(x,y)
print("Closed form: ",theta)

theta_gd=gradient_decent(x,y,0.0005,[0.1,0.1],300)
print("Gradient: ", theta_gd)
