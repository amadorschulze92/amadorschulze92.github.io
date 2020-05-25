---
layout: post
author: Michael A. Schulze
subtitle: Feature Importance
intro: Let's answer how to determine what clues your model is using to come to a solution?
---

# Feature Importance

```python
%run featimp
import pandas as pd
import numpy as np
```

More recently it was brought to my attention that the baked in feature importance methods in one of the more popular python machine leaning libraries was not entirely accurate. More specifically, the feature importance method in scikit-learn's Random Forest implementation was not working properly and was only partially fix last year. At a past job I used that feature importance method and made claims about how my model was predicting something to a sales person I interfaced with. After Tarrence Parr informed me of this mistake in scikit-learn's code, I became interested in how to compute these feature importances on my own so if I was ever unsure of the built in method I could check it with my own implementation. So in the cells below I will walk you through how I went about creating my simple python feature importance functions.

# Spearman's rank correlation coefficient

The idea behind this feature importance is pretty simple but the equation at the end has been known to scare people who commonly shirk math equations. The concept is comparing the rank of all the values in a feature column to the rank of all the values in the target column then using this equation...

$$\frac{6\Sigma d^2_{i}}{n(n^2-1)}$$

n = number of rows


d = the difference of each row's ranking between the feature and the target

Now here is my implementation...


```python
def spearman(df):
    df_new = df.copy()
    for feat in df.columns.tolist():
        df_new[f'{feat}_Rank'] = df[feat].rank(ascending=False, method='average')
    rank_cols = [x for x in df_new.columns.tolist() if '_Rank' in x]
    feat_spear_co = {}
    for feat in rank_cols:
        feat_name = feat.split()[0]
        feat_spear_co[feat] = abs(1-(6*sum((df_new[feat]-df_new["target_Rank"])**2)/(len(df)*(len(df)**2-1))))
    return feat_spear_co
```

Let's walk through this function one step at a time. I copy the dataframe just to prevent me from making changes to the original dataframe this should not happen because I am doing all my changes to the df_new, but thought I would leave in my supersticious code to remind myself of my coding mortality. Next I rank each column and input each columns' rank and the target's rank into the equation above. I save those results in a dictionary and return that dictionary as my output.

# Permutation Importance / Drop Column Importance

Next let's look at permutation and drop column importance another fairly simple method of computing the importance of a feature. Both are similar, but with a small twist in the implementation. Let's start with permutation. As the name suggests we use permutation of a particular column (aka we randomly shuffle the column) and then retrain our model and see if there was any change in our evaluation metric. If the feature is importance we should see a large decrease in our metric since an important variable used to predict our target hasa broken its relationship with the target variable. We repreat this process for all the independent variables and use those differences in our evaluation metric to determine the rank of importance for each feature. Drop column importance has a similar premise, we drop one column, retrain our model, then compare both models' evaluation metrics. We repeat this process for all the independent features and then rank which features missing from the model change the evaluation metric the most. Here is my implementation...


```python
def drop_importance(model, x, y_train, rounds):
    my_importances = {}
    for i in range(rounds):
        x_train = x.copy()
        x_train["random"] = np.random.randint(0,100,size=(len(x_train), 1))
        baseline = model.fit(x_train, y_train)
        baseline_score = baseline.score(x_train, y_train)
        for col in x_train.columns.tolist():
            x_train_drop = x_train.drop(columns=[col])
            drop = model.fit(x_train_drop, y_train)
            drop_score = drop.score(x_train_drop, y_train)
            my_importances[col] = baseline_score - drop_score
    my_importances = {k:np.mean(v) for k,v in my_importances.items()}
    return my_importances


def permute_importance(model, x, y_train, rounds):
    my_importances = {}
    for i in range(rounds):
        x_train = x.copy()
        x_train["random"] = np.random.randint(0,100,size=(len(x_train), 1))
        baseline = model.fit(x_train, y_train)
        baseline_score = baseline.score(x_train, y_train)
        for col in x_train.columns.tolist():
            x_train_permute = x_train.copy()
            x_train_permute[col] = np.random.permutation(x_train_permute[col])
            permute = model.fit(x_train_permute, y_train)
            permute_score = permute.score(x_train_permute, y_train)
            my_importances[col] = baseline_score - permute_score
    my_importances = {k:np.mean(v) for k,v in my_importances.items()}
    return my_importances
```

Again, let's step through this code together. First, I create a dictionary to store my eventual output. Next I repeat the permutation or drop process a number of times to make sure I reduce the noise in my data. What "noise" you ask? Good question, in my particular function I am also including a column of random values as a control for when I evaluate my features at the end and to make sure that random column is less likely to have an effect on my outcome I am looping my function for x number of times. This is more useful in my permutations implementation since the permutation part is random so by looping this part multiple times it should reduce the variance I would see im my answers. So now that I made my "random" column I create a baseline model that I will use later to caculate how much my model changes when a single feature is changed. Now I loop through each column and either permuting it or dropping it and recalculating the new evaluation metric. Lastly I calculate the difference between the baseline model evaluation and the new model's evaluation and save that value in the dictionary I use as my output.

Now that's all good and well, but how do I make decisions with those computed functions? I think this part is easier to explain with visuals so let's look at a regression dataset for california housing...

A linear model with drop importance

![lr_c_drop.png](attachment:lr_c_drop.png)

A linear model with permute importance

![lr_c_permute.png](attachment:lr_c_permute.png)

In these two cases the drop and permute had identical results, but you can see here that the random column had no impact on the model when dropped of permuted since it should not impact the model. Also it looks like when modeling this linearly the MedInc has a large impact on the price.

Now let's check out what our same drop and permute methods look like on the same data but trained using a random forest...

A random forest with drop importance

![rf_c_drop.png](attachment:rf_c_drop.png)

A random forest with permute importance

![rf_c_permute.png](attachment:rf_c_permute.png)

Again we see the drop and permute to have the same results, but this is not always the case. In another dataset I was testing with, the other variables differ slightly in the order the features appear in importance. Notice that with random as our control variable we can see that a number of columns in this model are not useful since they have a similar level of importance as the random column. You might be wondering why the features of importnace in these graphs differ from the features in our first two. The reason is because we are using two different models, linear regression and random forest, and discussing the differnces between those models are another article on its own. For those only looking for an easy explaination to be able to tell the sales person from down the hall, the models are seeing aspects in both sets of important features that carry valuable information in making a decision, but they make that decision differently and thus choose those variables differently.

# Comparing Feature Importance Metrics

So which feature importnace method should I use? Rather than just telling you "it depends" let's look at simple comparison graphically...

X axis = number of independent features (from most to least important)

Y axis = mean absolute error (mae)



blue = linear permute

green = spearman

red (hidden behind blue) = linear drop

![lr_c_eval.png](attachment:lr_c_eval.png)


```python
X axis = number of independent features (from most to least important)

Y axis = mean absolute error (mae)

blue = random forest permute

green = spearman

red (hidden behind blue) = random forest drop
```

![rf_c_eval.png](attachment:rf_c_eval.png)

So what do those pretty colors and lines mean? Well we want a lower MAE (aka we want to line to be lower on the y axis), the x axis is like if I kept adding the next most important feature/column and each line is the differnt methods for choosing the feature's importnace. We can see for the first graph (linear regression) spearman does much worse than permutation and drop but then for the second graph (random forest) that spearman is the best. So which feature importance gets you the best score depends on your data and on your model.

# Auto Feature Selection

Have you heard that some of the best programmers are also some of the laziest people? This is true, but the difference between a lazy person watching Netflix all day and a programmer is that the programmer is also always striving for efficiency. So why go through the trouble of making 15 different graphs and copying and pasting two blocks of my code when you can just make a function that finds the best features for you.


```python
def auto_feat_imp(model, feat_imp, x, y, x_val, y_val):
    last_mae = 100
    last_model = model.fit(x, y)
    c = Counter(feat_imp)
    for top_num in reversed(range(len(x.columns.tolist()))):
        top_cols = [k[0] for k in c.most_common(top_num+1) if k[0] != 'random']
        print(top_cols)
        m = model.fit(x[top_cols], y)
        mae = mean_absolute_error(y_val, m.predict(x_val[top_cols]))
        if last_mae < mae:
            break
        else:
            last_mae = mae
            last_model = m
    return last_model
```
