#%%
# -*- coding: utf-8 -*-
import numpy as np, matplotlib.pyplot as plt
import pymc3 as pm, theano.tensor as tt, theano
floatX = theano.config.floatX
# Libraries:
import pandas as pd
import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from IPython import display
#matplotlib inline
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#%%
# parse the dataset into three dataset: features (X), targets (y) and protected attributes (S)
def load_compas_data (path):
    input_data = (pd.read_csv(path))
    features = ['race', 'sex']
    X = (input_data.loc[:, features])
    X['race'] = X['race'].map({'African-American': 0,'Caucasian': 1, 'Hispanic': 2, 'Native American': 3,'Asian': 4, 'Other': 5})
    X['sex'] = X['sex'].map({'Female': 0,'Male': 1})
    y = input_data['two_year_recid']
    predictions = input_data['score_text']
    predictions = predictions.map({'Low': 0,'Medium': 1,'High': 1})
    return X, y, predictions

# load the train dataset
test_S, test_y, predictionsRaw = load_compas_data('data/compas-scores-two-years.csv')


protectedAttributes = test_S.values
originalPredictions = test_y.values
predictions = predictionsRaw.values

#%%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = 0.0 # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])-np.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))-np.log((1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%%
# Measure SP-Subgroup fairness (gamma unfairness) 
def subgroupFairness(probabilitiesOfPositive,alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = np.zeros(len(probabilitiesOfPositive)) # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*abs(spD-probabilitiesOfPositive[i])
    gamma = max(gammaPerGroup) # overall SF of the algorithm 
    return gamma

#%% Full Bayesian Methods
#%%
# hierarchical Bayesian logistic regression
def hierarchicalLR(test_S,predictions,numSamples,burnIn):
    data=pd.get_dummies(test_S.astype('category'))
    # one hot-encode of protected attributes
    protectedAttMask=np.int32((data.drop_duplicates()).values)
    #observed counts of instances, each entry corresponding to the corresponding row in protectedAttMask
    countsYis1 = np.zeros(len(protectedAttMask)) 
    countsYis0 = np.zeros(len(protectedAttMask)) 
    for i in range(len(data)):
        indx=np.where((protectedAttMask==data.values[i]).all(axis=1))[0][0]
        if predictions[i] == 1:
            countsYis1[indx] += 1
        else:
            countsYis0[indx] += 1
    countsTotal = countsYis1 + countsYis0 #total observed counts for each intersection of the protected attributes
    
    # fraction of group population
    population = len(predictions)
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(population + concentrationParameter)
    
    #hyperparameters
    mu = 0.0 #prior mean for logistic regression coefficients.
    sigma_1 = 1.0 #prior sd for logistic regression coefficients
    sigma_2_lambda = 10.0 #exponential distribution lambda (rate) parameter for hyperprior on sigma_2, the standard deviation for the deviances between the logistic regression predictions and the actual P(Y|S) probabilities
      
    #%%
    #Setup model
    model = pm.Model()
    with model:
        sigma_2 = pm.Exponential('sigma_2', lam  = sigma_2_lambda) #exponential distribution prior has a peak at 0, i.e. the deviance from the logistic regression model's predictions is 0.
        beta = pm.Normal('beta', mu = mu, sd = sigma_1, shape = protectedAttMask.shape[1]) #S_0 = 0, S_0 = 1, ... S_N = K. Logistic regression coefficients for each value of each of the protected attributes
        intercept = pm.Normal('intercept', mu = mu, sd = sigma_1) #intercept term in the logistic regression
        muGamma = pm.math.matrix_dot(protectedAttMask, beta) + intercept#the logistic regression model's predictions for the probability that Y=1, in logit space, for each intersection of the protected attributes.
        gamma = pm.Normal('gamma', mu = muGamma, sd = sigma_2, shape = protectedAttMask.shape[0]) #the probabilities for Y=1|S, in logit space, deviate from the logistic regression model's predictions with normally distributed noise
        p = pm.Deterministic('p_y_given_s', var=pm.math.sigmoid(gamma)) #P(Y=1|S) for each S
        counts = pm.Binomial('counts', p=p, n=countsTotal, observed=countsYis1) #instead of a Bernoulli for each instance's label, we equivalently have a Binomial for the counts of labels given S
        
    #%%
    #MAP estimation
    with model:
        map_estimate = pm.find_MAP()
    
    #%%
    #Fit model
# =============================================================================
#     with model:
#         tr = pm.sample(numSamples,chains=4, tune=burnIn)
# =============================================================================
    with model:
        inference = pm.ADVI()
        approx = pm.fit(burnIn, method=inference)
    with model:
        tr = approx.sample(draws=numSamples)
    
    #%%
    #Plot the output of the model
    #pm.plots.traceplot(tr, ['p_y_given_s']);
    #pm.plots.traceplot(tr, ['sigma_2']);
    probabilitiesOfPositive=(tr['p_y_given_s']).mean(axis=0)
    mcmcSamples = tr['p_y_given_s']
    return probabilitiesOfPositive,alphaG_smoothed,mcmcSamples,protectedAttMask

prob_harLR,alpha_harLR,Samples_harLR,data_harLR = hierarchicalLR(test_S,predictions,1000,200000)
np.savetxt('data_harLR.txt',data_harLR)
np.savetxt('Samples_harLR.txt',Samples_harLR)
np.savetxt('prob_harLR.txt',prob_harLR)
np.savetxt('alpha_harLR.txt',alpha_harLR)

def bayesianLR(test_S,predictions,numSamples,burnIn):
    data=pd.get_dummies(test_S.astype('category'))
    # one hot-encode of protected attributes
    protectedAttMask=np.int32((data.drop_duplicates()).values)
    #observed counts of instances, each entry corresponding to the corresponding row in protectedAttMask
    countsYis1 = np.zeros(len(protectedAttMask)) 
    countsYis0 = np.zeros(len(protectedAttMask)) 
    for i in range(len(data)):
        indx=np.where((protectedAttMask==data.values[i]).all(axis=1))[0][0]
        if predictions[i] == 1:
            countsYis1[indx] += 1
        else:
            countsYis0[indx] += 1
    countsTotal = countsYis1 + countsYis0 #total observed counts for each intersection of the protected attributes
    
    # fraction of group population
    population = len(predictions)
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(population + concentrationParameter)
    
    #hyperparameters
    mu = 0.0 #prior mean for logistic regression coefficients.
    sigma_1 = 1.0 #prior sd for logistic regression coefficients
    
    #%%
    #Setup model
    model = pm.Model()
    with model:
        beta = pm.Normal('beta', mu = mu, sd = sigma_1, shape = protectedAttMask.shape[1]) #S_0 = 0, S_0 = 1, ... S_N = K. Logistic regression coefficients for each value of each of the protected attributes
        intercept = pm.Normal('intercept', mu = mu, sd = sigma_1) #intercept term in the logistic regression
        logits = pm.math.matrix_dot(protectedAttMask, beta) + intercept#the logistic regression model's predictions for the probability that Y=1, in logit space, for each intersection of the protected attributes.
        p = pm.Deterministic('p_y_given_s', var=pm.math.sigmoid(logits)) #P(Y=1|S) for each S
        counts = pm.Binomial('counts', p=p, n=countsTotal, observed=countsYis1) #instead of a Bernoulli for each instance's label, we equivalently have a Binomial for the counts of labels given S
        
#%%
    #MAP estimation
    with model:
        map_estimate = pm.find_MAP()
    
# =============================================================================
#     with model:
#         tr = pm.sample(numSamples,chains=4, tune=burnIn)
# =============================================================================
    with model:
        inference = pm.ADVI()
        approx = pm.fit(burnIn, method=inference)
    with model:
        tr = approx.sample(draws=numSamples)
    
    #%%
    #Plot the output of the model
    #pm.plots.traceplot(tr, ['p_y_given_s']);
    #pm.plots.traceplot(tr, ['sigma_2']);
    probabilitiesOfPositive=(tr['p_y_given_s']).mean(axis=0)
    mcmcSamples = tr['p_y_given_s']
    return probabilitiesOfPositive,alphaG_smoothed,mcmcSamples,protectedAttMask

prob_LR_fb,alpha_LR_fb,Samples_LR_fb,data_LR_fb = bayesianLR(test_S,predictions,1000,200000)
np.savetxt('data_LR_fb.txt',data_LR_fb)
np.savetxt('Samples_LR_fb.txt',Samples_LR_fb)
np.savetxt('prob_LR_fb.txt',prob_LR_fb)
np.savetxt('alpha_LR_fb.txt',alpha_LR_fb)
#%%
# Bayesian naive Bayes
def bayesianNB(test_S,test_y,numSamples,burnIn):
    numObservations = 2
    numRace = 6
    numGender = 2
    y_probs = np.zeros(numObservations)
    S1_probs = np.zeros([numObservations,numRace])
    S2_probs = np.zeros([numObservations,numGender])
    #S3_probs = np.zeros([numObservations,numNationality])
    
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(test_y)):
        y_probs[test_y[i]] = y_probs[test_y[i]] + 1
        S1_probs[test_y[i],test_S['race'][i]] = S1_probs[test_y[i],test_S['race'][i]] + 1
        S2_probs[test_y[i],test_S['sex'][i]] = S2_probs[test_y[i],test_S['sex'][i]] + 1
        
    y_probs = y_probs/len(test_y)
    S1_probs=(S1_probs+dirichletAlpha)/(S1_probs+concentrationParameter).sum(axis=1)[:,np.newaxis]
    S2_probs=(S2_probs+dirichletAlpha)/(S2_probs+concentrationParameter).sum(axis=1)[:,np.newaxis]
    
    #%%
    # set up the Bayesian naive Bayes model
    NB_model = pm.Model()
    
    with NB_model:
        obs_probs = pm.Categorical('obs_probs',p=y_probs,shape=1)
        
        S1_probs = theano.shared(S1_probs)
        S1_probs_chosen = S1_probs[obs_probs]
        s1 = pm.Categorical('s1',p=S1_probs_chosen)
        
        S2_probs = theano.shared(S2_probs)
        S2_probs_chosen = S2_probs[obs_probs]
        s2 = pm.Categorical('s2',p=S2_probs_chosen)
    
    with NB_model:
        trace_NB = pm.sample(numSamples,chains=4,tune=burnIn)
    
    yPositive = np.reshape(trace_NB['obs_probs']==1,(len(trace_NB['obs_probs']),))
    
    # calculating positive probabilities from samples generated by Bayesian naive Bayes model
    S1=test_S['race'].unique() # 0,1,2,3 values represent Black, White, Asian-Pac-Islander and Other races respectively
    S2=test_S['sex'].unique() # 0,1 values represent Female and Male respectively
    #S3=test_S['nationality'].unique() # 0,1 values represent USA and other nationalities respectively
    
    probabilitiesOfPositive = np.zeros(len(S1)*len(S2))
    data = np.zeros((len(S1)*len(S2),2))
    alphaG_smoothed = np.zeros(len(probabilitiesOfPositive)) # fraction of group population
    protectedAttributes = test_S.values
    predictions = test_y
    
    # fraction of group population
    # compute counts and probabilities
    population = len(predictions)
    countsClassOne = np.zeros((6,6)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsClassTwo = np.zeros((6,6))
    countsTotal = np.zeros((6,6))
    
    for i in range(len(predictions)):
        countsTotal[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        
        if predictions[i] == 1:   # class one
            countsClassOne[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        else:
            countsClassTwo[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
            
    indx = 0
    for i in S1:
        for j in S2:
            data[indx,0]=i
            data[indx,1]=j
            temp = np.logical_and(trace_NB['s1']==i,trace_NB['s2']==j)
            probabilitiesOfPositive[indx]=np.sum(np.logical_and(yPositive,temp))/np.sum(temp)
            alphaG_smoothed[indx] = (countsTotal[i,j] + dirichletAlpha) /(population + concentrationParameter)
            indx += 1
    
    alphaG_smoothed=alphaG_smoothed[probabilitiesOfPositive>0]
    probabilitiesOfPositive=probabilitiesOfPositive[probabilitiesOfPositive>0]
    alphaG_smoothed=alphaG_smoothed[probabilitiesOfPositive<1] 
    probabilitiesOfPositive=probabilitiesOfPositive[probabilitiesOfPositive<1]    
    return probabilitiesOfPositive,alphaG_smoothed,data

prob_NB_fb,alpha_NB_fb,data_NB_fb = bayesianNB(test_S,predictions,1000,60000)
np.savetxt('data_NB_fb.txt',data_NB_fb)
np.savetxt('prob_NB_fb.txt',prob_NB_fb)
np.savetxt('alpha_NB_fb.txt',alpha_NB_fb)
#%%
# Bayesian neural network
def bayesianNN(test_S, predictions,n,numSamples):
    
    data=pd.get_dummies(test_S.astype('category'))
    # one hot-encode of protected attributes
    protectedAttMask=np.int32((data.drop_duplicates()).values)
    #observed counts of instances, each entry corresponding to the corresponding row in protectedAttMask
    countsYis1 = np.zeros(len(protectedAttMask)) 
    countsYis0 = np.zeros(len(protectedAttMask)) 
    for i in range(len(data)):
        indx=np.where((protectedAttMask==data.values[i]).all(axis=1))[0][0]
        if predictions[i] == 1:
            countsYis1[indx] += 1
        else:
            countsYis0[indx] += 1
    countsTotal = countsYis1 + countsYis0 #total observed counts for each intersection of the protected attributes
    
    # fraction of group population
    population = len(predictions)
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(population + concentrationParameter)
    
    n_hidden = 10

    # Initialize random weights between each layer
    init_1 = np.random.randn(protectedAttMask.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)
    
    neural_network = pm.Model()
    with neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(protectedAttMask.shape[1], n_hidden),
                                 testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                  shape=(n_hidden,),
                                  testval=init_out)

        # Build neural-network using tanh activation function
# =============================================================================
#         act_1 = pm.math.tanh(pm.math.dot(protectedAttMask,
#                                          weights_in_1))
#         act_2 = pm.math.tanh(pm.math.dot(act_1,
#                                          weights_1_2))
# =============================================================================
        act_1 = tt.nnet.relu(pm.math.dot(protectedAttMask,
                                         weights_in_1))
        act_2 = tt.nnet.relu(pm.math.dot(act_1,
                                         weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2,
                                              weights_2_out))

        # Binary classification -> Bernoulli likelihood
        p = pm.Deterministic('p_y_given_s', var=act_out) #P(Y=1|S) for each S
        counts = pm.Binomial('counts', p=p, n=countsTotal, observed=countsYis1)
        
    #MAP estimation
    with neural_network:
        map_estimate = pm.find_MAP()
    #map_estimate    
    #%%    
    with neural_network:
        inference = pm.ADVI()
        approx = pm.fit(n, method=inference)
    with neural_network:
        tr = approx.sample(draws=numSamples)
    
    probabilitiesOfPositive=(tr['p_y_given_s']).mean(axis=0)
    mcmcSamples = tr['p_y_given_s']
    
    return probabilitiesOfPositive,alphaG_smoothed,mcmcSamples,protectedAttMask

prob_NN_fb,alpha_NN_fb,Samples_NN_fb,data_NN_fb = bayesianNN(test_S, predictions,200000,1000)
np.savetxt('data_NN_fb.txt',data_NN_fb)
np.savetxt('Samples_NN_fb.txt',Samples_NN_fb)
np.savetxt('prob_NN_fb.txt',prob_NN_fb)
np.savetxt('alpha_NN_fb.txt',alpha_NN_fb)

#%%
# original samples
Oprob_harLR,Oalpha_harLR,OSamples_harLR,Odata_harLR = hierarchicalLR(test_S,originalPredictions,1000,200000)
np.savetxt('Odata_harLR.txt',Odata_harLR)
np.savetxt('OSamples_harLR.txt',OSamples_harLR)
np.savetxt('Oprob_harLR.txt',Oprob_harLR)
np.savetxt('Oalpha_harLR.txt',Oalpha_harLR)

Oprob_LR_fb,Oalpha_LR_fb,OSamples_LR_fb,Odata_LR_fb = bayesianLR(test_S,originalPredictions,1000,200000)
np.savetxt('Odata_LR_fb.txt',Odata_LR_fb)
np.savetxt('OSamples_LR_fb.txt',OSamples_LR_fb)
np.savetxt('Oprob_LR_fb.txt',Oprob_LR_fb)
np.savetxt('Oalpha_LR_fb.txt',Oalpha_LR_fb)

Oprob_NB_fb,Oalpha_NB_fb,Odata_NB_fb = bayesianNB(test_S,originalPredictions,1000,60000)
np.savetxt('Odata_NB_fb.txt',Odata_NB_fb)
np.savetxt('Oprob_NB_fb.txt',Oprob_NB_fb)
np.savetxt('Oalpha_NB_fb.txt',Oalpha_NB_fb)

Oprob_NN_fb,Oalpha_NN_fb,OSamples_NN_fb,Odata_NN_fb = bayesianNN(test_S, originalPredictions,200000,1000)
np.savetxt('Odata_NN_fb.txt',Odata_NN_fb)
np.savetxt('OSamples_NN_fb.txt',OSamples_NN_fb)
np.savetxt('Oprob_NN_fb.txt',Oprob_NN_fb)
np.savetxt('Oalpha_NN_fb.txt',Oalpha_NN_fb)

#%%
# Point Estimate methods + Full Bayesian Method

#%%
# empirical differential fairness measurement
# Note that EDF is a point estimate model, EDF-Smooth is a Bayesian model
def computeEDFforData(protectedAttributes,predictions):
    # compute counts and probabilities
    countsClassOne = np.zeros((6,6)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsClassTwo = np.zeros((6,6))
    countsTotal = np.zeros((6,6))
    
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(predictions)):
        countsTotal[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        
        if predictions[i] == 1:   # class one
            countsClassOne[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        else:
            countsClassTwo[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
    
    probabilitiesClassOne = countsClassOne/countsTotal
    probabilitiesClassOneSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    S1 = np.unique(protectedAttributes[:,0]) # number of races
    S2 = np.unique(protectedAttributes[:,1]) # number of gender
    #S3 = np.unique(protectedAttributes[:,2]) # number of nationalities
    probabilitiesForDF = np.zeros(len(S1)*len(S2))
    data = np.zeros((len(S1)*len(S2),2))
    probabilitiesForDFSmoothed = np.zeros(len(probabilitiesForDF))
    alphaG = np.zeros(len(probabilitiesForDF)) #
    alphaG_smoothed = np.zeros(len(probabilitiesForDF))
    population = len(predictions)
    indx = 0
    for i in S1:
        for j in S2:
            data[indx,0]=i
            data[indx,1]=j
            probabilitiesForDF[indx] = probabilitiesClassOne[i,j]
            probabilitiesForDFSmoothed[indx] = probabilitiesClassOneSmoothed[i,j]
            alphaG_smoothed[indx] = (countsTotal[i,j] + dirichletAlpha) /(population + concentrationParameter)
            alphaG[indx] = (countsTotal[i,j]) /(population)
            indx += 1
                
    alphaG=alphaG[probabilitiesForDF>0]
    probabilitiesForDF=probabilitiesForDF[probabilitiesForDF>0]
    alphaG=alphaG[probabilitiesForDF<1] 
    probabilitiesForDF=probabilitiesForDF[probabilitiesForDF<1]     
    return probabilitiesForDF,alphaG,probabilitiesForDFSmoothed,alphaG_smoothed,data

prob_EDF,alpha_EDF,prob_EDFSmoothed,alpha_EDFSmoothed,data_EDF = computeEDFforData(protectedAttributes,predictions)
np.savetxt('prob_EDF.txt',prob_EDF)
np.savetxt('alpha_EDF.txt',alpha_EDF)
np.savetxt('prob_EDFSmoothed.txt',prob_EDFSmoothed)
np.savetxt('alpha_EDFSmoothed.txt',alpha_EDFSmoothed)
np.savetxt('data_EDF.txt',data_EDF)

#%%
# Point Estimate methods

#%%
# BernoulliNB is implemented which should suitable for discrete data
def naiveBayesModel(protectedAttributes,predictions):
    clfNB = BernoulliNB(alpha=.1,binarize=0.0)
    clfNB.fit(protectedAttributes,predictions)
    S1 = np.unique(protectedAttributes[:,0]) # number of races
    S2 = np.unique(protectedAttributes[:,1]) # number of gender
    probabilitiesOfPositive = np.zeros(len(S1)*len(S2))
    data = np.zeros((len(S1)*len(S2),2))
    alphaG_smoothed = np.zeros(len(probabilitiesOfPositive)) # fraction of group population
    
    # maintain count to get fraction of group population
    # compute counts and probabilities
    population = len(predictions)
    countsClassOne = np.zeros((6,6)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsClassTwo = np.zeros((6,6))
    countsTotal = np.zeros((6,6))
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    for i in range(len(predictions)):
        countsTotal[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        if predictions[i] == 1:   # class one
            countsClassOne[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        else:
            countsClassTwo[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
    
    
    indx = 0
    for i in S1:
        for j in S2:
            data[indx,0]=i
            data[indx,1]=j
            probabilitiesOfPositive[indx]=clfNB.predict_proba([[i,j]])[0,1]
            alphaG_smoothed[indx] = (countsTotal[i,j] + dirichletAlpha) /(population + concentrationParameter)
            indx += 1
                

    return probabilitiesOfPositive,alphaG_smoothed,data
prob_NB,alpha_NB,data_NB = naiveBayesModel(protectedAttributes,predictions)
np.savetxt('prob_NB.txt',prob_NB)
np.savetxt('alpha_NB.txt',alpha_NB)
np.savetxt('data_NB.txt',data_NB)

#%%
# Logistic Regression to model p(y|S)
def logisticRegressionModel(protectedAttributes,predictions):
    clfLR = LogisticRegression(C=1.0,solver='liblinear')
    clfLR.fit(protectedAttributes,predictions)
    S1 = np.unique(protectedAttributes[:,0]) # number of races
    S2 = np.unique(protectedAttributes[:,1]) # number of gender
    probabilitiesOfPositive = np.zeros(len(S1)*len(S2))
    data = np.zeros((len(S1)*len(S2),2))
    alphaG_smoothed = np.zeros(len(probabilitiesOfPositive)) # fraction of group population
    
    # maintain count to get fraction of group population
    # compute counts and probabilities
    population = len(predictions)
    countsClassOne = np.zeros((6,6)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsClassTwo = np.zeros((6,6))
    countsTotal = np.zeros((6,6))
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    for i in range(len(predictions)):
        countsTotal[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        if predictions[i] == 1:   # class one
            countsClassOne[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        else:
            countsClassTwo[protectedAttributes[i,0],protectedAttributes[i,1]] += 1

    indx = 0
    for i in S1:
        for j in S2:
            data[indx,0]=i
            data[indx,1]=j
            probabilitiesOfPositive[indx]=clfLR.predict_proba([[i,j]])[0,1]
            alphaG_smoothed[indx] = (countsTotal[i,j] + dirichletAlpha) /(population + concentrationParameter)
            indx += 1
                
    return probabilitiesOfPositive,alphaG_smoothed,data
prob_LR,alpha_LR,data_LR = logisticRegressionModel(protectedAttributes,predictions)
np.savetxt('prob_LR.txt',prob_LR)
np.savetxt('alpha_LR.txt',alpha_LR)
np.savetxt('data_LR.txt',data_LR)
#%%
# deep neural networks using sci-kit learn
def neuralNetworkModel(protectedAttributes,predictions):
    clfDNN = MLPClassifier(hidden_layer_sizes=(10,10,10,),activation='relu',solver='adam',random_state=0)
    clfDNN.fit(protectedAttributes,predictions)
    S1 = np.unique(protectedAttributes[:,0]) # number of races
    S2 = np.unique(protectedAttributes[:,1]) # number of gender
    probabilitiesOfPositive = np.zeros(len(S1)*len(S2))
    data = np.zeros((len(S1)*len(S2),2))
    alphaG_smoothed = np.zeros(len(probabilitiesOfPositive)) # fraction of group population
    
    # maintain count to get fraction of group population
    # compute counts and probabilities
    population = len(predictions)
    countsClassOne = np.zeros((6,6)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsClassTwo = np.zeros((6,6))
    countsTotal = np.zeros((6,6))
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    for i in range(len(predictions)):
        countsTotal[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        if predictions[i] == 1:   # class one
            countsClassOne[protectedAttributes[i,0],protectedAttributes[i,1]] += 1
        else:
            countsClassTwo[protectedAttributes[i,0],protectedAttributes[i,1]] += 1

    indx = 0
    for i in S1:
        for j in S2:
            data[indx,0]=i
            data[indx,1]=j
            probabilitiesOfPositive[indx]=clfDNN.predict_proba([[i,j]])[0,1]
            alphaG_smoothed[indx] = (countsTotal[i,j] + dirichletAlpha) /(population + concentrationParameter)
            indx += 1
                
    return probabilitiesOfPositive,alphaG_smoothed,data
prob_NN,alpha_NN,data_NN  = neuralNetworkModel(protectedAttributes,predictions)
np.savetxt('prob_NN.txt',prob_NN)
np.savetxt('alpha_NN.txt',alpha_NN)
np.savetxt('data_NN.txt',data_NN)

#%% original predictions
Oprob_EDF,Oalpha_EDF,Oprob_EDFSmoothed,Oalpha_EDFSmoothed,Odata_EDF = computeEDFforData(protectedAttributes,originalPredictions)
np.savetxt('Oprob_EDF.txt',Oprob_EDF)
np.savetxt('Oalpha_EDF.txt',Oalpha_EDF)
np.savetxt('Oprob_EDFSmoothed.txt',Oprob_EDFSmoothed)
np.savetxt('Oalpha_EDFSmoothed.txt',Oalpha_EDFSmoothed)
np.savetxt('Odata_EDF.txt',Odata_EDF)

Oprob_NB,Oalpha_NB,Odata_NB = naiveBayesModel(protectedAttributes,originalPredictions)
np.savetxt('Oprob_NB.txt',Oprob_NB)
np.savetxt('Oalpha_NB.txt',Oalpha_NB)
np.savetxt('Odata_NB.txt',Odata_NB)

Oprob_LR,Oalpha_LR,Odata_LR = logisticRegressionModel(protectedAttributes,originalPredictions)
np.savetxt('Oprob_LR.txt',Oprob_LR)
np.savetxt('Oalpha_LR.txt',Oalpha_LR)
np.savetxt('Odata_LR.txt',Odata_LR)

Oprob_NN,Oalpha_NN,Odata_NN  = neuralNetworkModel(protectedAttributes,originalPredictions)
np.savetxt('Oprob_NN.txt',Oprob_NN)
np.savetxt('Oalpha_NN.txt',Oalpha_NN)
np.savetxt('Odata_NN.txt',Odata_NN)

