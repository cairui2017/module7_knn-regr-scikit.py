import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def main():
    N = int(input("Enter N (positive integer): "))
    k = int(input("Enter k (positive integer): "))
    
    # Check if k is less than or equal to N
    if k > N:
        print("Error: k must be less than or equal to N.")
        return
    
    points = np.zeros((N, 2)) # Initialize a N x 2 numpy array to store the points
    
    for i in range(N):
        x = float(input(f"Enter x value for point {i+1}: "))
        y = float(input(f"Enter y value for point {i+1}: "))
        points[i] = [x, y]
    
    X_input = float(input("Enter X: "))
    
    # Split the points into X and y for training
    X_train = points[:, 0].reshape(-1, 1) # Reshape for a single feature
    y_train = points[:, 1]
    
    # Create and train the k-NN model
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predict the output for the input X
    Y_pred = knn.predict([[X_input]])
    
    # Calculate the coefficient of determination (R^2)
    y_pred_full = knn.predict(X_train)
    r2 = r2_score(y_train, y_pred_full)
    
    print(f"The result (Y) for X={X_input} using k-NN Regression is: {Y_pred[0]}")
    print(f"The coefficient of determination (R^2): {r2}")

if __name__ == "__main__":
    main()
