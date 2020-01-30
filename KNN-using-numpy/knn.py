
class Knn:
# k-Nearest Neighbor class object for classification training and testing
    def __init__(self):
        self.X_train = np.array(())
        self.y_train = np.array(())
        
    def fit(self, x, y):
        # Save the training data to properties of this class
        self.X_train = x.to_numpy()
        self.y_train = y.to_numpy()
        
    def predict(self, x, k):
        y_hat = [] # Variable to store the estimated class label for 
        # Calculate the distance from each vector in x to the training data
        x = x.to_numpy()
        for i in range(len(x)):
            dist_Xx = np.sqrt(np.sum((self.X_train-np.array(x[i]))**2, axis=1))
            near_neigh_k = np.argsort(dist_Xx)[:k]
            near_neigh = self.y_train[near_neigh_k]
            #predicted_y = np.bincount(near_neigh).argmax()  #For only integers y values
            val,count = np.unique(near_neigh, return_counts=True)
            predicted_y = val[np.argmax(count)]
            y_hat.append(predicted_y)
        # Return the estimated targets
            
        return np.array(y_hat)

# Metric of overall classification accuracy
#  (a more general function, sklearn.metrics.accuracy_score, is also available)
def accuracy(y,y_hat):
    nvalues = len(y)
    accuracy = sum(y == y_hat) / nvalues
    return accuracy
