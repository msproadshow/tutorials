import matplotlib.pyplot as plt
import numpy as np
import cntk as C
from sklearn.model_selection import train_test_split

def transform_data(data, target, test_size=0.3):
    
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

    def map(X,Y):
        num_classes = len(np.unique(Y))
        Y = np.vectorize(lambda y: np.array([y]), signature='()->(m)')(Y)
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
        X = X.astype(np.float32)
        return X,Y
    
    x_train, y_train = map(x_train, y_train)
    x_test, y_test  = map( x_test, y_test )
    
    return x_train, y_train, x_test, y_test

def create_params(input_var, output_dim):
    input_dim = input_var.shape[0]
    
    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))
    
    return weight_param, bias_param

def create_learner(z, learning_rate = 0.5):
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch) 
    return C.sgd(z.parameters, lr_schedule)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    from matplotlib.colors import ListedColormap 
    y = np.argmax(y, axis=1)
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')    
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')    
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1    
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))    

    Z = np.argmax(classifier.eval(np.array([xx1.ravel(), xx2.ravel()]).T), axis=1)
    Z = Z.reshape(xx1.shape)    
    
    fig = plt.figure(figsize=(8,5))
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())    
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples    
    X_test, y_test = X[test_idx, :], y[test_idx]                                   
    for idx, cl in enumerate(np.unique(y)):        
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples    
    if test_idx:        
        X_test, y_test = X[test_idx, :], y[test_idx]           
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

def draw_iris(iris):

    feature_names, label_names = iris.feature_names, iris.target_names
    features, labels = iris.data, iris.target
    
    colors = ['r' if label == 0 else 'b' if label == 1 else 'g' for label in labels]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

    ax1.scatter(features[:,0], features[:,1], c=colors)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])

    ax2.scatter(features[:,2], features[:,3], c=colors)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])

    plt.show()


def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


def update_training_progress(trainer, iteration, frequency, plotdata, verbose=1):

    training_loss = trainer.previous_minibatch_loss_average
    eval_error = trainer.previous_minibatch_evaluation_average

    plotdata["iteration"].append(iteration)
    plotdata["loss"].append(training_loss)
    plotdata["error"].append(eval_error)

    if verbose and (iteration % frequency == 0): 
        print ("Epoch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(iteration, training_loss, eval_error))
        
    return plotdata

def plot_learning_process(plotdata):
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plotdata["avgerror"] = moving_average(plotdata["error"])

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["iteration"], plotdata["avgloss"], 'b--')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.title('Epoch run vs. Training loss')

    plt.show()

    plt.subplot(212)
    plt.plot(plotdata["iteration"], plotdata["avgerror"], 'r--')
    plt.xlabel('Epoch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Epoch run vs. Label Prediction Error')
    plt.show()