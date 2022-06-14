# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:59:42 2022

@author: novar
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

class LSTMlayer:
    def forward(self, x_t, h_prev, c_prev, params):
        self.c_prev = c_prev
        self.z = np.row_stack((h_prev,x_t))
        self.f = sigmoid(np.dot(params["Wf"],self.z)+params["bf"])
        self.i = sigmoid(np.dot(params["Wi"],self.z)+params["bi"])
        self.c_ = np.tanh(np.dot(params["Wc"],self.z)+params["bc"])
        self.c = self.f*self.c_prev + self.i*self.c_
        self.o = sigmoid(np.dot(params["Wo"],self.z)+params["bo"])
        self.h = self.o*np.tanh(self.c)
        self.v = np.dot(params["Wv"],self.h) + params["bv"]
        self.y_t = softmax(self.v)
        return self.c, self.h, self.y_t
        
    def backward(self, params, y, dh_next, dc_next):
        #run forward first for creating self parameters
        #daXXX; a denotes activaiton
        dv = self.y_t.copy() - y
        grads_step = {}
        grads_step["dWv"] = np.dot(dv, self.h.T)
        grads_step["dbv"] = dv
        dh = np.dot(params["Wv"].T, dv)
        dh += dh_next

        do = dh*np.tanh(self.c)
        da_o = do*self.o*(1-self.o)
        grads_step["dWo"] = np.dot(da_o, self.z.T)
        grads_step["dbo"] = da_o

        dc = dh*self.o*(1-np.square(np.tanh(self.c)))
        dc += dc_next

        dc_bar = dc * self.i
        da_c = dc_bar * (1-np.square(self.c))
        grads_step["dWc"] = np.dot(da_c, self.z.T)
        grads_step["dbc"] = da_c

        di = dc*self.c_
        da_i = di*self.i*(1-self.i) 
        grads_step["dWi"] = np.dot(da_i, self.z.T)
        grads_step["dbi"] = da_i

        df = dc*self.c_prev
        da_f = df*self.f*(1-self.f)
        grads_step["dWf"] = np.dot(da_f, self.z.T)
        grads_step["dbf"] = da_f

        dz = (np.dot(params["Wf"].T, da_f) + np.dot(params["Wi"].T, da_i)
             + np.dot(params["Wc"].T, da_c) + np.dot(params["Wo"].T, da_o))
    
        dh_prev = dz[:self.h.shape[0], :]
        dc_prev = self.f * dc
        return dh_prev, dc_prev, grads_step
# DONE


class LSTM:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        rHx = np.sqrt(1/(self.Lxdim+self.Lhid))
        Lshape = (self.Lhid,self.Lxdim+self.Lhid)
        #forget gate
        self.params = {}
        self.params["Wf"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bf"] = np.ones((self.Lhid,1))
        #input gate
        self.params["Wi"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bi"] = np.ones((self.Lhid,1))
        #cell gate
        self.params["Wc"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bc"] = np.ones((self.Lhid,1))
        #output gate
        self.params["Wo"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bo"] = np.ones((self.Lhid,1))
        #output
        self.params["Wv"] = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lclass,self.Lhid))
        self.params["bv"] = np.ones((self.Lclass,1))
        #grads
        self.grads = {}
        self.gradsNew = {}
        for key in self.params:
            self.grads["d"+key] = np.zeros(self.params[key].shape)
            self.gradsNew["d"+key] = np.zeros(self.params[key].shape)
    
    def forward(self, data):
        # data is (T,) size timeseries, parellelize after implementing single steps
        T = len(data)
        hidden, c = np.zeros((self.Lhid,1)), np.zeros((self.Lhid,1))
        foldedLayers = []
        for t in range(T):
            layer = LSTMlayer()
            hidden, c, self.outProb = layer.forward(data[t][:,None], hidden, c, self.params)
            foldedLayers.append(layer)
        self.outProb = self.outProb.T # transpose for convention
        self.Layers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        # out is OneHot encoded
        self.out = np.zeros(self.Lclass)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forward before to update self.outProb
        assert ground.shape == self.outProb.shape
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        #run after forward
        lyr = self.Layers
        t = self.Lfeature - 1
        #
        dh_next, dc_next, grads_step = lyr[t].backward(self.params, ground.T, np.zeros((self.Lhid,1)), np.zeros((self.Lhid,1)))
        for key in self.gradsNew:
            self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # change t with (self.Lfeature - 1) if you want
            y_t = lyr[i].y_t
            dh_next, dh_c, grads_step = lyr[i].backward(self.params, y_t, dh_next, dc_next) # input y_t so that dWv is zero
            for key in self.gradsNew:
                self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        for key in self.params:
            self.grads['d'+key] = momentum*self.grads['d'+key] + self.gradsNew['d'+key]
            self.params[key] = self.params[key] - learningRate*self.grads['d'+key]
            self.gradsNew['d'+key] = np.zeros(self.params[key].shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target)
        self.calcGrad(sample, target)
        return loss, guess

def trainMiniBatch(nnModel, data, ground, valX, valD, testX, testD, epoch, learningRate, momentum, batchSize = 32):
    countSamples = 0
    lossListT, lossListV, accuracyListT, accTest= [], [], [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        
        estimatesT = []
        loss = 0
        for j in range(len(grounds)):
            bSize = grounds[j].shape[0]
            for i in range(bSize):
                countSamples += 1
                l, g = nnModel.trainStep(samples[j][i], grounds[j][i][None,:])
                estimatesT.append(g)
                loss += l
            nnModel.updateWeights(learningRate, momentum)
        loss = loss/totalSamples
        lossListT.append(loss)
        
        gndidx = np.array([np.where(r==1)[0][0] for r in shuffled_grounds]) + 1
        estidx = np.array([np.where(r==1)[0][0] for r in estimatesT]) + 1
        
        falses = np.count_nonzero(gndidx-estidx)
        accuracy = 1-falses/totalSamples
        accuracyListT.append(accuracy)
        
        loss = 0
        for i in range(valD.shape[0]):
            nnModel.forward(valX[i])
            guess, _ = nnModel.forwardOut()
            loss += nnModel.crossEntropy(valD[i][None,:])
        loss = loss/valD.shape[0]
        lossListV.append(loss)
        
        estTest = []
        for i in range(testD.shape[0]):
            nnModel.forward(testX[i])
            guess, _ = nnModel.forwardOut()
            estTest.append(guess)
        
        Tgndidx = np.array([np.where(r==1)[0][0] for r in testD]) + 1
        estTestidx = np.array([np.where(r==1)[0][0] for r in estTest]) + 1
        
        falses = np.count_nonzero(Tgndidx-estTestidx)
        accuracy = 1-falses/testD.shape[0]
        accTest.append(accuracy)
        
        print(f"Validation Loss in epoch {e+1}: {loss}, Test Accuracy: {accuracy}")
        if loss > 1.2*lossListV[0]: 
            print("Termnated due to increased loss")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.0001):
            print("Terminated due to convergence")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
    return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def comp_confmat(actual, predicted):
    np.seterr(divide='ignore')
    classes = np.unique(actual)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confmat 

def plotTwinParameter(metric, labels):
    xlabel = [i for i in range(len(metric[0]))]
    plt.plot(xlabel, metric[0], marker='o', markersize=6, linewidth=2, label=labels[0])
    plt.legend()
    plt.ylabel(labels[0])
    plt.ylim((0,1.1))
    ax2 = plt.twinx()
    ax2.plot(xlabel, metric[1], marker='o', color = 'red', markersize=6, linewidth=2, label=labels[1])
    plt.ylabel(labels[1])
    plt.title('Parameter vs Metrics Plot')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotParameter(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [i for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with Learning Rate: {metricName[2]}, Momentum: {metricName[3]}, BPTT: {metricName[4]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotConf(mat_con, Title):
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m,y=n,s=int(mat_con[m, n]), va='center', ha='center', size='xx-large')
    
    # Sets the labels
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix for '+Title, fontsize=15)
    plt.show()
# In[Read the data]
filename = "data3.h5"

with h5py.File(filename, "r") as f:
    groupKeys = list(f.keys())
    sets = []
    for key in groupKeys:
        sets.append(list(f[key]))
del key
# In[]
idx = np.random.permutation(3000)
trainX = np.array(sets[0])[idx]
trainD = np.array(sets[1])[idx]
testX = np.array(sets[2])
testD = np.array(sets[3])
valX = trainX[:300]
valD = trainD[:300]
trainX = trainX[300:]
trainD = trainD[300:]
# In[]
bptt = 3
model = LSTM(150, 3, 128, 6, bptt)
model.initializeWeights()
lossT, lossV, accT, accTest = [], [], [], []
# In[]
lr = 0.01
mm = 0.85
epoch = 10
print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
lossT.extend(l1)
lossV.extend(l2) 
accT.extend(a1)
accTest.extend(a2)
# In[]
plotConf(confT, "Training Set, RNN")
plotConf(confTest, "Test Set, RNN")
# In[plot]
plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","LSTM",lr,mm,bptt])
#%%
plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","LSTM",lr,mm,bptt])