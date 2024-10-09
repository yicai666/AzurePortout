#scp -P 8030 -F ./sshconfig /Users/yc010e/Documents/DS_summit_2024/retail_df.csv yc010e@att.com@135.170.35.237:/tmp
"""
Downsample 
"""
def downsample(df, classratio):
    DP = df[df['fraud'].astype(int) == 1 ]
    DN = df[df['fraud'].astype(int) == 0 ].sample(n=int(DP.shape[0]*classratio))
    D = pd.concat([DP, DN]).sample(frac=1.0)
    return D
# train = downsample(train,downsampleRatio)

def sampling(df,over_sampling_strategy,under_sampling_strategy):
    X,y = df.drop('fraud',axis = 1), df['fraud']
    over = SMOTENC(sampling_strategy = over_sampling_strategy, random_state = 42)
    under = RandomUnderSampler(sampling_strategy = under_sampling_strategy)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)
    return(X, y)


def data_split(df,channel,catColumns, numColumns, method,split_date = '2023-12-10', random_down_rate = 0.05, smote_up_rate = 0.5, sampling = 'downsample'):
    channel_df = df[df['channel'].isin(channel)].copy()
    channel_df[catColumns] = channel_df[catColumns].astype("category") 
    if method == 'random':
        train, test = train_test_split(channel_df, test_size=0.2, random_state=423,stratify = channel_df['fraud'])
    elif method == 'time':
        train = channel_df[channel_df['load_date']<=split_date]
        test = channel_df[channel_df['load_date']>split_date]

    predictors = catColumns + numColumns 
    X_train = train[predictors]
    X_test = test[predictors]
    y_train = train['fraud'].astype('int')
    y_test = test['fraud'].astype('int')

    if sampling == 'downsample':
        under = RandomUnderSampler(sampling_strategy = random_down_rate)
        X_train, y_train = under.fit_resample(X_train,y_train)
    if sampling == 'smote':
        under = RandomUnderSampler(sampling_strategy = random_down_rate)
        over = SMOTENC(categorical_features = 'auto',random_state = 42,sampling_strategy = smote_up_rate)
        steps = [('u', under),('o', over)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

    return train,test,X_train,X_test,y_train,y_test

def score(params):
    # mlflow.lightgbm.autolog()
    # with mlflow.start_run(nested=True):
        #print("Training with params: ")
        #print(params)
    if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
    if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
    if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
    
    estimator = LGBMClassifier(objective='binary', 
                            random_state=314, 
                            silent=True,
                            metric = 'average_precision')

    model = estimator.set_params(**params) #model.named_steps['classifier'].set_params(**params)

    # cross validation
    shuffle = KFold(n_splits=4, shuffle=True)
    perf_score = cross_val_score(model, X_train, y_train, cv=shuffle, scoring='average_precision', n_jobs=-1)
    avg_score = perf_score.mean()
    std_score = perf_score.std()

    # mlflow.log_metric('avg_pr', avg_score)
    # mlflow.log_metric('std_pr',std_score)
    loss = -1 * avg_score

    print(f"mean pr-auc: {avg_score}")
    return {'loss':loss , 'status': STATUS_OK}

def lgbm_train(tuning_model_name, final_model_name,tune = True, save_model = False,model_path=None):

    #### Step 1: hyperparameter tuning
    # estimator = LGBMClassifier(objective='binary', 
    #                             eval_metric='average_precision',
    #                             silent = True,
    #                             random_state = 1234)
    # model = estimator
        
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    scale = weights[1]/weights[0]
    #print('pos/neg ratio is',scale)

    if tune == True:
        search_space = {
        'num_leaves': 10,
        'max_depth' : hp.quniform('max_depth', 2, 6,1),          # depth of trees or  use scorpe.int()
        'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.15)),
        'min_gain_to_split': hp.quniform('min_gain_to_split', 0.1, 0.5, 0.05),  # minimum loss reduction required to make a further partition on a leaf node   
        'min_child_weight' : hp.quniform('min_child_weight', 10, 30, 5),   # minimum number of instances per node
        'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),       # random selection of rows for training,       
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05), # proportion of columns to use per tree
        'colsample_bynode' : hp.quniform('colsample_bynode', 0.5, 1, 0.05), # proportion of columns to use per node
        'scale_pos_weight' : hp.quniform('scale_pos_weight', 10, 120, 10),
        # 'scale_pos_weight' : hp.quniform('scale_pos_weight', scale/10, scale*1.2, 10),  
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 400, 50)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        #'learning_rate': hp.quniform('eta', 0.001, 0.5, 0.02),
        #'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        #'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        #'gamma': hp.quniform('gamma', 0.5, 2, 0.05),
        #'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        #'scale_pos_weight': scope.int(hp.loguniform('scale_pos_weight', np.log(1), np.log(scale_pos_weight_max)))
        }

        trials = Trials()
        with mlflow.start_run(run_name=tuning_model_name):
            argmin = fmin(
            fn = score,
            space=search_space,
            algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
            max_evals = 200,
            trials= trials, 
            rstate=np.random.default_rng(123),
            verbose=False)
        
        params = space_eval(search_space, argmin)

    else:
        params = {
            'num_leaves': 10,
            'colsample_bynode': 0.3832441567595607,
            'colsample_bytree': 0.14653178537132974,
            'min_gain_to_split': 0.896,
            'learning_rate': 0.03188180105192845,
            'max_depth': 5,
            'min_child_weight': 20,
            'n_estimators': 300,
            'reg_alpha': 0.06675773265651277,
            'reg_lambda': 0.8660860794936944,
            'scale_pos_weight': 30,
            'subsample': 0.5582568771220421,
            'verbose_eval': True}

    ### Step 2: final model training
    mlflow.lightgbm.autolog()
    with mlflow.start_run(run_name= final_model_name) as run:
    
    # capture run info for later use
        #run_id = run.info.run_id
        # run_name = run.data.tags['mlflow.runName']
        # run_ids += [(run_name, run_id)]
    
    # configure params
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
        if 'min_child_weight' in params:
            params['min_child_weight']=int(params['min_child_weight'])
        if 'max_delta_step' in params: 
            params['max_delta_step']=int(params['max_delta_step'])
        if 'scale_pos_weight' in params:
            params['scale_pos_weight']=int(params['scale_pos_weight'])    
            # params['tree_method']='hist'        # modified for CPU deployment
            # params['predictor']='cpu_predictor' # modified for CPU deployment
        mlflow.log_params(params)
    
    # train
        model = LGBMClassifier(objective = 'binary',
                        random_state=314, 
                        silent=True, 
                        metric='average_precision')
        model.set_params(**params)
        model.fit(X_train, y_train, 
                #   early_stopping_rounds = 10, 
                #  eval_metric = 'average_precision', 
                #   eval_set = [(X_test,y_test)],
                #   eval_names = ['valid'],
                #   verbose = 50,
                  feature_name = 'auto', # that's actually the default
                  categorical_feature = 'auto')
        #mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
    
    # # predict
        y_prob = model.predict_proba(X_test)[:,1]
    
    # score
        model_ap = average_precision_score(y_test, y_prob)
        model_auc = roc_auc_score(y_test, y_prob)
        print(f"test set pr-auc: {model_ap}")
        mlflow.log_metric('avg precision', model_ap)
        mlflow.log_metric('auc', model_auc)
        if save_model:
            mlflow.sklearn.save_model(model, model_path)
    
        #print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))
    return(model,y_prob,params)

def perf_metrics(df,pert_list,pert,score_version):
    threshold = pert_list[pert]
    df['pred_cat'] = (df[score_version]>threshold)*1
    precision, recall, fscore, support = pfscore(df['fraud'], df['pred_cat'])
    actual_sending_rate = np.round((df['pred_cat'].mean())*100,1)
    CM = confusion_matrix(df['fraud'], df['pred_cat'])
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
#     print('High risk group percentile: {}'.format(eva_df['pred_cat'].mean()))
#     print('confusion matrix: \n {}'.format(pd.crosstab( eva_df['fraud'],eva_df['pred_cat'])))
#     print('precision: {}'.format(precision))
#     print('recall: {}'.format(recall))
#     print('support: {}'.format(support))
    metrix_array = np.array([np.round(pert,4),
                   np.round(threshold,7),
                   np.round(actual_sending_rate,4),
                   TP,
                   FP,
                   np.round(precision[1],4),
                   np.round(recall[1],3),
                   np.round(fscore[1],3)])
    return(metrix_array)

## the input is the data set with 'fraud' as the fraud indicator and score_version as the model score name
def perf_array(eva_df, score_version):
    threshold_list = eva_df[score_version].quantile([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9,0.85,0.8,0.75])
    metrics = []
    for pert in threshold_list.index:
        line = perf_metrics(eva_df,threshold_list,pert,score_version)
        metrics = np.concatenate((metrics,line),axis=0)
        
    metrics.shape = (len(threshold_list),8)
    metrics_array = metrics.transpose()

    d = {'cumulative_data_fraction': (1-metrics_array[0])*100,
     'threshold':metrics_array[1],'actual_sending_rate':metrics_array[2],
     'TP':metrics_array[3],'FP':metrics_array[4],
     'Precision':metrics_array[5],'Recall':metrics_array[6],
     'F1':metrics_array[7]}
    df = pd.DataFrame(data=d)  
    df = df[['cumulative_data_fraction','threshold','actual_sending_rate','TP','FP','Precision','Recall','F1']]
    df = df.sort_values(by='cumulative_data_fraction', ascending=True)
    df['cumulative_data_fraction'] = df['cumulative_data_fraction'].astype(int).astype(str)
    df['cumulative_data_fraction'] = df['cumulative_data_fraction']+'%'
    df['actual_sending_rate'] = df['actual_sending_rate'].astype(str)
    df['actual_sending_rate'] = df['actual_sending_rate']+'%'
    
    df = df[['actual_sending_rate','threshold','TP','FP','Precision','Recall','F1']]
    df.style.hide_index()
    
    return(df)

def precison_recall_metrics(label,pred):
    fpr, tpr, thresh = metrics.roc_curve(label, pred)
    auc = np.round(metrics.roc_auc_score(label, pred),3)
    precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
    average_precision = np.round(metrics.average_precision_score(label, pred),3)
    #prauc = np.round(metrics.auc(precision,recall),3)
    prauc = np.round(average_precision_score(label,pred),3)
    f1 = 2*precision*recall/(precision + recall)
    return fpr, tpr, thresholds, auc, precision,recall,f1,prauc


def perf_comp(score_df, model_name):

    ################################################################# 
    # input df : ['applicationId','fraud','score1','score2','score3',...]
    # model name: model name 
    #################################################################

    for i in [col for col in score_df.columns if col != 'applicationId']:
        score_df[i] = score_df[i].astype(float)
    score_df = score_df.dropna()
    n_comparison = score_df.shape[1] - 2
    print(f'number of models is {n_comparison}')
    d = {}
    perf_df = pd.DataFrame()

    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    fig.tight_layout()
    for i in range(n_comparison):
        index = i + 1
        d[f"label{index}"] = score_df['fraud']
        d[f'score{index}' ] = score_df[f'score{index}']

        # precison_recall_metrics(d['label{0}'. ],d['score{0}'. ])
        d[f'fpr{index}' ], d[f'tpr{index}' ], d[f'thresh{index}'],d[f'auc{index}'],d[f'precision{index}'],d[f'recall{index}'],d[f'f1_{index}'],d[f'aucpr{index}'] = precison_recall_metrics(d[f'label{index}'],d[f'score{index}'])

        # plt.subplot(1, 2, 1)
        ax[0].plot(d[f'fpr{index}'],d[f'tpr{index}'],label=  f"{model_name[i]},auc="+ str(d[f'auc{index}']))
        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')
        ax[0].set_title('ROC Curve Comparison')
        ax[0].legend(loc=4)
        
        # plt.subplot(1, 2, 2)
        ax[1].plot(d[f'recall{index}'],d[f'precision{index}'],label= f"{model_name[i]},AUCPR ="+ str(d[f'aucpr{index}']))
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set_title('Precision-Recall Curve Comparison')
        plt.legend(loc=0)

        perf = perf_array(score_df,f'score{index}')
        if index > 1:
            perf = perf.drop('actual_sending_rate',axis = 1)
        perf = perf.add_suffix(f'_{model_name[i]}')
        perf_df = pd.concat([perf_df,perf], axis = 1)

    plt.show()

    return(perf_df)

