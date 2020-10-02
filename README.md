# Python-project - An Awesome Project
#Cost function 
def compute_cost(X, y, theta):
    J = 0
    A = np.subtract(np.dot(X, theta), y)
    J = (0.5) * (np.dot(A.T, A))
    return J[0, 0]


#gradient descent

def gradient_descent(X, y, theta, alpha, epsilon, flag):
    J_hist = []
    all_thetas0 = []
    all_thetas1 = []
    bt_val = 0
    while (1):
        gradient = alpha * np.dot(X.T, np.subtract(np.dot(X, theta), y))
        cost_curr = compute_cost(X, y, theta)
        # print(cost_curr)
        J_hist.append(cost_curr)
        all_thetas0.append(theta[0])
        all_thetas1.append(theta[1])
        if (len(J_hist) >= 2):
            if (math.fabs(J_hist[len(J_hist) - 1] - J_hist[len(J_hist) - 2]) <
                    epsilon):
                break
        if (len(J_hist) >= 2):
            if (math.fabs(J_hist[len(J_hist) - 1] - J_hist[len(J_hist) - 2]) >
                    9999):
                bt_val = 1
                break
        theta = theta - gradient
    if (flag == 1):
        plt.figure(2)
        plt.plot(J_hist)
        plt.xlabel('Number of Iterations -->')
        plt.ylabel('Value of the Cost function(J) -->')
        plt.title('Checking convergence with alpha : ' + str(float(alpha)))
        plt.show(block=False)
        raw_input("Hit Enter To Close")
        plt.close()
    return theta, np.array(all_thetas0), np.array(all_thetas1), np.array(
        J_hist), bt_val



#main
def main(x_training, y_training):
    x_training = x_training.T
    y_training = y_training.T


    #plot the initial training data
    print('\n\nPlotting the data...\n')
    plt.figure(1)
    plt.plot(x_training, y_training, 'ro')
    plt.xlabel('Acidity -->')
    plt.ylabel('Density -->')
    plt.title('Training Set')
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()


    #Design matrix X
    x_training = (x_training - np.mean(x_training)) / (
        np.std(x_training))  #feature scaling using mean normalization
    # x_training = (x_training - np.mean(x_training)) / (np.max(x_training) - np.mean(x_training))
    m = y_training.shape[0]  #number of training examples
    X = np.hstack((np.reshape(np.ones(m), (m, 1)), x_training))
    theta = np.reshape(np.zeros(2), (2, 1))  #0-initialization of theta


    #some gradient descent settings
    epsilon = math.pow(10, -10)
    alpha = 0.0007


    print('\nTesting the cost function ...\n')
    #compute and display the initial cost
