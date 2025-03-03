from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm 
import numpy as np
from collections import Counter
from sklearn.neighbors import KernelDensity
from datetime import datetime
import os



title_set = 'original'
save_folder = '../experiment/graphs'

n_class_A = 40 
n_class_B = 160

point_size = int(72*0.5)



def create_dataset(n_samples,weights,n_classes,n_features):


    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=n_classes,
                           n_clusters_per_class=1,
                           weights=weights,
                           class_sep=0.3, random_state=0)
    
    mapping = {max(y): 'B'}
    for cls in y:
        if cls not in mapping:
            mapping[cls] = 'A'

    encoded_classes = np.array([mapping[cls] for cls in y])
    rows = np.hstack((encoded_classes.reshape(-1, 1),X))
    dataset = pd.DataFrame(rows,columns = ['Class','at1','at2'])

    return X,y,dataset


def create_dataset_norm(n_a,n_b):

    np.random.seed(42)
    # Parameters for the normal distributions
    mean1 = [2, 2]  # Mean of the first distribution
    cov1 = [[1, 0.5], [0.5, 1]]  # Covariance matrix for first group (slightly correlated)
    mean2 = [4, 4]  # Mean of the second distribution (shifted slightly to create overlap)
    cov2 = [[1, 0.5], [0.5, 1]]  # Covariance matrix for second group (similar structure)
    # Generate 12 points for the first distribution
    groupA = np.random.multivariate_normal(mean1, cov1, n_a) # class A
    # Generate 8 points for the second distribution
    groupB = np.random.multivariate_normal(mean2, cov2, n_b) # class B
    #stack arrays horizontally
    data = np.vstack([groupA,groupB])
    class_a = np.array([0 for i in range(n_a)])  # Class labels for Class (A)  0
    class_b = np.array([1 for j in range(n_b)]) # Class labels for Class (B)   1
    # Concatenate the class labels
    classes = np.concatenate((class_a, class_b))
    print(classes)

    rows = np.hstack([classes.reshape(-1, 1),data])
    dataset = pd.DataFrame(rows,columns = ['Class','at1','at2'])


    
    return data, classes, dataset

def occurrence_def(Y):
# Count occurrences of each element
    counts = Counter(Y)
    # Find the most frequently occurring element and its count
    max_occurring = max(counts, key=counts.get)
    min_occurring = min(counts,key=counts.get)
    max_value = counts[max_occurring]
    min_value = counts[min_occurring]
    IR = {}
    lista = []
    for classe,count in counts.items():
        value = max_value-count
        lista.append((classe,value))
        imb_ratio = count/min_value
        IR[classe] = round(imb_ratio,3)
        
    print(f'Imbalance Ratio (IR)  {IR}')
    return sorted(lista, key=lambda x: x[1], reverse=False), sorted(IR.items(), key=lambda pair: pair[1], reverse=True), max_value


def kde_sampler(kernel,data,n_istances):
    kde = KernelDensity(kernel = kernel, algorithm = 'ball_tree', bandwidth = 'silverman').fit(data)
    examples = kde.sample(n_istances, random_state=0)

    return examples



def stacking_def(original_array,label,array):
    #print('New_data:',array.shape)
    labels = [ label for i in range(len(array))]
    #print('New labels:',len(labels))
    rows = np.vstack([labels,array.T]).T
    return np.vstack([original_array,rows])



def oversamp_KDE_definitive(X,Y):
    #ranges_kde, ranges_original = {},{} 
    #ranges_original = check_range(data,ranges_original,classe)
    #call function to provide minority classes and missing examples for each to match majoirty class
    lista, IR, max_value = occurrence_def(Y)
    print(f'Imbalance Ratio (IR)  {IR}')
    for item in IR:
        if item[1] > 1:
                imbalance =  True
                break
    
    dataset = np.vstack([Y,X.T]).T
    print('Original dataset', dataset.shape)
    if imbalance:
        stacking_array =  dataset
        print('\n') 
        print(f'Majority class num of istances: {max_value}')
        for item in lista[1:]:
          classe, n_istances = item[0], item[1]
        #selecting minority examples BY INDEX
          indices = [i for i, class_value in enumerate(Y) if class_value == classe]
          data = X[indices,:]
          print('Selected minority:',data.shape)
        #creating density estimation and sampling new data
          examples = kde_sampler('gaussian',data,n_istances)
          stacking_array = stacking_def(stacking_array,classe,examples)
          print('KDE - Class: ', classe, ';NEW istances generated:', len(examples))
          print('New exampels: ',examples.shape)
          #print('Sampled EXAMPLES:\n',examples)     
          print('Updated dataframe:',stacking_array.shape,'\n')

    #ranges_kde = check_range(examples,ranges_kde, classe)
    print('FINAL OVERSAMPLED dataframe:',stacking_array.shape)
    #tupling class num to new examples
    x = stacking_array[:,1:]
    y = stacking_array[:,0].astype(int)
    print('Oversampled with KDE:', Counter(y))

    return examples,x,y



def plot_dataset(x, y,a_data,b_data, sampler, result_date):

        
    plt.figure(figsize=(12, 12))
    if sampler == 'Over':
        x_new,x_res,y_res = oversamp_KDE_definitive(x,y)
        title_set = 'Oversampled dataset - {} new istances'.format((len(x_new)))

    #A in red, B in blue , A new in green
        plt.scatter(a_data[:, 0], a_data[:, 1],  alpha=0.8, color = 'red',edgecolor="k",s=point_size)
        plt.scatter(b_data[:, 0], b_data[:, 1],  alpha=0.8, color = 'blue',edgecolor="k", s=point_size)
        plt.scatter(x_new[:,0],x_new[:,1], alpha = 0.8, color = 'green',edgecolor="k", s=point_size)
    else:
        title_set = f"Original dataset"
        plt.scatter(a_data[:, 0], a_data[:, 1], alpha=0.8, color = 'red',edgecolor="k",s=point_size)
        plt.scatter(b_data[:, 0], b_data[:, 1], alpha=0.8, color = 'blue',edgecolor="k",s=point_size)

    plt.title(title_set)
    #sns.despine(ax=ax, offset=10)
    file_name = os.path.join(save_folder, '{}_{}'.format(result_date,sampler))
    plt.savefig(file_name)
    plt.close()


#Plot the KDE for the dataset: one graph with the two KDE (one for each class)

def kde_countour2D(x,y ,bandwidth, cmap, xbins=100j, ybins=100j, **kwargs):
    # create grid of sample locations (default: 100x100)
    xmin, xmax = -1, 8
    ymin, ymax = -1, 8
    xx, yy = np.mgrid[xmin:xmax:xbins, ymin:ymax:ybins]

    #Build 2D kernel density estimate (KDE)
    train_data = np.vstack([y, x]).T
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    kde_model = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_model.fit(train_data)
    # score_samples() returns the log-likelihood of the samples
    scores = np.exp(kde_model.score_samples(xy_sample))
    zz = np.reshape(scores, xx.shape)

    plt.contourf(xx, yy, zz, cmap=cmap, alpha = 0.5)


    

#Plot on top of the dataset the new 4 points generated by KDE to balance the dataset (now 24 points). Highlight these 4 points in green.
#Plot the new KDE for the balanced dataset.


    
#dataset,x,y = create_dataset(20,[0.40],2,2)

x,y,dataset = create_dataset_norm(n_class_A,n_class_B)
print(dataset)
    
    
result_date = datetime.now().strftime("%Y%m%d_%H%M")

a_index = [i for i, class_value in enumerate(y) if class_value == 0] #class A
b_index = [i for i, class_value in enumerate(y) if class_value == 1]  #class B
a_data, b_data = x[a_index,:], x[b_index,:]
print(a_data)
print(b_data)
    
plot_dataset(x, y,a_data,b_data, 'Over', result_date)
plot_dataset(x, y,a_data,b_data, None,result_date)


    
fig1 = plt.gcf()
ax = fig1.gca()
plt.figure(figsize=(12, 12))

plt.title('Original')
plt.scatter(a_data[:,0], a_data[:,1], facecolor = 'blue', s=point_size)
plt.scatter(b_data[:,0],b_data[:,1], facecolor  = 'red', s=point_size)


kde_countour2D(a_data[:,0], a_data[:,1],'silverman', cmap=cm.Blues)
kde_countour2D(b_data[:,0], b_data[:,1],'silverman', cmap=cm.Reds)

file_name = os.path.join(save_folder, '{}_KDE_ORI'.format(result_date))
plt.savefig(file_name)
plt.close()

    #---------------------------------------------------------------
examples,x_res,y_res = oversamp_KDE_definitive(x,y)

a_index = [i for i, class_value in enumerate(y_res) if class_value == 0] #class A
b_index = [i for i, class_value in enumerate(y_res) if class_value == 1]  #class B
a_data_ov, b_data_ov = x_res[a_index,:], x_res[b_index,:]

fig2 = plt.gcf()
ax = fig2.gca()
plt.figure(figsize=(12, 12))

plt.title('Oversampled dataset - {} new istances'.format(int(n_class_B-n_class_A)))
plt.scatter(a_data_ov[:,0], a_data_ov[:,1], facecolor = 'blue', s=point_size)
plt.scatter(b_data_ov[:,0],b_data_ov[:,1], facecolor  = 'red', s=point_size)


kde_countour2D(a_data_ov[:,0], a_data_ov[:,1],'silverman', cmap=cm.Blues)
kde_countour2D(b_data_ov[:,0], b_data_ov[:,1],'silverman', cmap=cm.Reds)

file_name = os.path.join(save_folder, '{}_KDE_OVSAP'.format(result_date))
plt.savefig(file_name)
plt.close()
