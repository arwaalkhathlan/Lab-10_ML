# Support Vector Machines 

Welcome to your SVM assignment! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data
For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

Here's a picture of the three different Iris types:


```python
# # The Iris Setosa
from IPython.display import Image, display
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
display(Image(url=url, width=300, height=300))
```


<img src="http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" width="300" height="300"/>



```python
# The Iris Versicolor
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
display(Image(url=url, width=300, height=300))
```


<img src="http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" width="300" height="300"/>



```python
# The Iris Virginica
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
display(Image(url=url, width=300, height=300))
```


<img src="http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" width="300" height="300"/>


The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)

The four features of the Iris dataset:

    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm

## Get the data

**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris')**


```python
import seaborn as sns
sns.set_style('darkgrid')
iris = sns.load_dataset('iris')
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the data and get you started!

## Exploratory Data Analysis

Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!

**Import some libraries you think you'll need.**


```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

**Create a pairplot of the data set. Which flower species seems to be the most separable?**


```python
sns.pairplot(iris, hue = 'species')
plt.show()
```


    
![png](output_9_0.png)
    


**Create a kde plot of sepal_length versus sepal width for setosa species of flower.**


```python
setosa = iris[iris["species"] == "setosa"]

sns.kdeplot(
    data=setosa,
    x="sepal_length",
    y="sepal_width",
    cmap="plasma",
    fill=True,
)
plt.show()
```


    
![png](output_11_0.png)
    


# Train Test Split

**Split your data into a training set and a testing set.**


```python
from sklearn.model_selection import train_test_split
```


```python
X = iris.drop('species', axis = 1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
```

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**


```python
from sklearn.svm import SVC
```


```python
mySVC = SVC()
```


```python
mySVC.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SVC</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>SVC()</pre></div> </div></div></div></div>



## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
y_predictions = mySVC.predict(X_test)
```


```python
print(confusion_matrix(y_test, y_predictions))
```

    [[13  0  0]
     [ 0 19  1]
     [ 0  0 12]]
    


```python
print(classification_report(y_test, y_predictions))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.95      0.97        20
       virginica       0.92      1.00      0.96        12
    
        accuracy                           0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    
    

Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

## Gridsearch Practice

**Import GridsearchCV from SciKit Learn.**


```python
from sklearn.model_selection import GridSearchCV
```

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**


```python
param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
```

**Create a GridSearchCV object and fit it to the training data.**


```python
myGS = GridSearchCV(SVC(), param_grid = param_grid, verbose = 3)
myGS.fit(X_train, y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV 1/5] END ....................C=0.1, gamma=1;, score=0.905 total time=   0.0s
    [CV 2/5] END ....................C=0.1, gamma=1;, score=1.000 total time=   0.0s
    [CV 3/5] END ....................C=0.1, gamma=1;, score=0.905 total time=   0.0s
    [CV 4/5] END ....................C=0.1, gamma=1;, score=0.905 total time=   0.0s
    [CV 5/5] END ....................C=0.1, gamma=1;, score=0.952 total time=   0.0s
    [CV 1/5] END ..................C=0.1, gamma=0.1;, score=0.857 total time=   0.0s
    [CV 2/5] END ..................C=0.1, gamma=0.1;, score=0.714 total time=   0.0s
    [CV 3/5] END ..................C=0.1, gamma=0.1;, score=0.857 total time=   0.0s
    [CV 4/5] END ..................C=0.1, gamma=0.1;, score=0.810 total time=   0.0s
    [CV 5/5] END ..................C=0.1, gamma=0.1;, score=0.810 total time=   0.0s
    [CV 1/5] END .................C=0.1, gamma=0.01;, score=0.714 total time=   0.0s
    [CV 2/5] END .................C=0.1, gamma=0.01;, score=0.714 total time=   0.0s
    [CV 3/5] END .................C=0.1, gamma=0.01;, score=0.714 total time=   0.0s
    [CV 4/5] END .................C=0.1, gamma=0.01;, score=0.714 total time=   0.0s
    [CV 5/5] END .................C=0.1, gamma=0.01;, score=0.714 total time=   0.0s
    [CV 1/5] END ................C=0.1, gamma=0.001;, score=0.333 total time=   0.0s
    [CV 2/5] END ................C=0.1, gamma=0.001;, score=0.333 total time=   0.0s
    [CV 3/5] END ................C=0.1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 4/5] END ................C=0.1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 5/5] END ................C=0.1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 1/5] END ...............C=0.1, gamma=0.0001;, score=0.333 total time=   0.0s
    [CV 2/5] END ...............C=0.1, gamma=0.0001;, score=0.333 total time=   0.0s
    [CV 3/5] END ...............C=0.1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 4/5] END ...............C=0.1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 5/5] END ...............C=0.1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 1/5] END ......................C=1, gamma=1;, score=1.000 total time=   0.0s
    [CV 2/5] END ......................C=1, gamma=1;, score=0.905 total time=   0.0s
    [CV 3/5] END ......................C=1, gamma=1;, score=0.905 total time=   0.0s
    [CV 4/5] END ......................C=1, gamma=1;, score=0.952 total time=   0.0s
    [CV 5/5] END ......................C=1, gamma=1;, score=0.952 total time=   0.0s
    [CV 1/5] END ....................C=1, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 2/5] END ....................C=1, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 3/5] END ....................C=1, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 4/5] END ....................C=1, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 5/5] END ....................C=1, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 1/5] END ...................C=1, gamma=0.01;, score=0.952 total time=   0.0s
    [CV 2/5] END ...................C=1, gamma=0.01;, score=0.810 total time=   0.0s
    [CV 3/5] END ...................C=1, gamma=0.01;, score=0.857 total time=   0.0s
    [CV 4/5] END ...................C=1, gamma=0.01;, score=0.857 total time=   0.0s
    [CV 5/5] END ...................C=1, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 1/5] END ..................C=1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 2/5] END ..................C=1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 3/5] END ..................C=1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 4/5] END ..................C=1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 5/5] END ..................C=1, gamma=0.001;, score=0.714 total time=   0.0s
    [CV 1/5] END .................C=1, gamma=0.0001;, score=0.333 total time=   0.0s
    [CV 2/5] END .................C=1, gamma=0.0001;, score=0.333 total time=   0.0s
    [CV 3/5] END .................C=1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 4/5] END .................C=1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 5/5] END .................C=1, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 1/5] END .....................C=10, gamma=1;, score=1.000 total time=   0.0s
    [CV 2/5] END .....................C=10, gamma=1;, score=0.905 total time=   0.0s
    [CV 3/5] END .....................C=10, gamma=1;, score=0.905 total time=   0.0s
    [CV 4/5] END .....................C=10, gamma=1;, score=0.905 total time=   0.0s
    [CV 5/5] END .....................C=10, gamma=1;, score=0.952 total time=   0.0s
    [CV 1/5] END ...................C=10, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 2/5] END ...................C=10, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 3/5] END ...................C=10, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 4/5] END ...................C=10, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 5/5] END ...................C=10, gamma=0.1;, score=0.952 total time=   0.0s
    [CV 1/5] END ..................C=10, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 2/5] END ..................C=10, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 3/5] END ..................C=10, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 4/5] END ..................C=10, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 5/5] END ..................C=10, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 1/5] END .................C=10, gamma=0.001;, score=0.952 total time=   0.0s
    [CV 2/5] END .................C=10, gamma=0.001;, score=0.810 total time=   0.0s
    [CV 3/5] END .................C=10, gamma=0.001;, score=0.857 total time=   0.0s
    [CV 4/5] END .................C=10, gamma=0.001;, score=0.857 total time=   0.0s
    [CV 5/5] END .................C=10, gamma=0.001;, score=0.905 total time=   0.0s
    [CV 1/5] END ................C=10, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 2/5] END ................C=10, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 3/5] END ................C=10, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 4/5] END ................C=10, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 5/5] END ................C=10, gamma=0.0001;, score=0.714 total time=   0.0s
    [CV 1/5] END ....................C=100, gamma=1;, score=1.000 total time=   0.0s
    [CV 2/5] END ....................C=100, gamma=1;, score=0.905 total time=   0.0s
    [CV 3/5] END ....................C=100, gamma=1;, score=0.857 total time=   0.0s
    [CV 4/5] END ....................C=100, gamma=1;, score=0.905 total time=   0.0s
    [CV 5/5] END ....................C=100, gamma=1;, score=0.952 total time=   0.0s
    [CV 1/5] END ..................C=100, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 2/5] END ..................C=100, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 3/5] END ..................C=100, gamma=0.1;, score=0.857 total time=   0.0s
    [CV 4/5] END ..................C=100, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 5/5] END ..................C=100, gamma=0.1;, score=0.952 total time=   0.0s
    [CV 1/5] END .................C=100, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 2/5] END .................C=100, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 3/5] END .................C=100, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 4/5] END .................C=100, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 5/5] END .................C=100, gamma=0.01;, score=0.952 total time=   0.0s
    [CV 1/5] END ................C=100, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 2/5] END ................C=100, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 3/5] END ................C=100, gamma=0.001;, score=0.905 total time=   0.0s
    [CV 4/5] END ................C=100, gamma=0.001;, score=0.905 total time=   0.0s
    [CV 5/5] END ................C=100, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 1/5] END ...............C=100, gamma=0.0001;, score=0.952 total time=   0.0s
    [CV 2/5] END ...............C=100, gamma=0.0001;, score=0.810 total time=   0.0s
    [CV 3/5] END ...............C=100, gamma=0.0001;, score=0.857 total time=   0.0s
    [CV 4/5] END ...............C=100, gamma=0.0001;, score=0.857 total time=   0.0s
    [CV 5/5] END ...............C=100, gamma=0.0001;, score=0.905 total time=   0.0s
    [CV 1/5] END ...................C=1000, gamma=1;, score=1.000 total time=   0.0s
    [CV 2/5] END ...................C=1000, gamma=1;, score=0.905 total time=   0.0s
    [CV 3/5] END ...................C=1000, gamma=1;, score=0.857 total time=   0.0s
    [CV 4/5] END ...................C=1000, gamma=1;, score=0.905 total time=   0.0s
    [CV 5/5] END ...................C=1000, gamma=1;, score=0.952 total time=   0.0s
    [CV 1/5] END .................C=1000, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 2/5] END .................C=1000, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 3/5] END .................C=1000, gamma=0.1;, score=0.857 total time=   0.0s
    [CV 4/5] END .................C=1000, gamma=0.1;, score=0.905 total time=   0.0s
    [CV 5/5] END .................C=1000, gamma=0.1;, score=0.952 total time=   0.0s
    [CV 1/5] END ................C=1000, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 2/5] END ................C=1000, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 3/5] END ................C=1000, gamma=0.01;, score=0.857 total time=   0.0s
    [CV 4/5] END ................C=1000, gamma=0.01;, score=0.905 total time=   0.0s
    [CV 5/5] END ................C=1000, gamma=0.01;, score=0.952 total time=   0.0s
    [CV 1/5] END ...............C=1000, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 2/5] END ...............C=1000, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 3/5] END ...............C=1000, gamma=0.001;, score=0.905 total time=   0.0s
    [CV 4/5] END ...............C=1000, gamma=0.001;, score=0.905 total time=   0.0s
    [CV 5/5] END ...............C=1000, gamma=0.001;, score=0.952 total time=   0.0s
    [CV 1/5] END ..............C=1000, gamma=0.0001;, score=1.000 total time=   0.0s
    [CV 2/5] END ..............C=1000, gamma=0.0001;, score=1.000 total time=   0.0s
    [CV 3/5] END ..............C=1000, gamma=0.0001;, score=0.905 total time=   0.0s
    [CV 4/5] END ..............C=1000, gamma=0.0001;, score=0.905 total time=   0.0s
    [CV 5/5] END ..............C=1000, gamma=0.0001;, score=1.000 total time=   0.0s
    




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=SVC(),
             param_grid={&#x27;C&#x27;: [0.1, 1, 10, 100, 1000],
                         &#x27;gamma&#x27;: [1, 0.1, 0.01, 0.001, 0.0001]},
             verbose=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(estimator=SVC(),
             param_grid={&#x27;C&#x27;: [0.1, 1, 10, 100, 1000],
                         &#x27;gamma&#x27;: [1, 0.1, 0.01, 0.001, 0.0001]},
             verbose=3)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: SVC</div></div></label><div class="sk-toggleable__content fitted"><pre>SVC(C=1, gamma=0.1)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SVC</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SVC(C=1, gamma=0.1)</pre></div> </div></div></div></div></div></div></div></div></div>



**Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**


```python
y_predictions = myGS.predict(X_test)
```


```python
print(confusion_matrix(y_test, y_predictions))
```

    [[13  0  0]
     [ 0 19  1]
     [ 0  0 12]]
    


```python
print(classification_report(y_test, y_predictions))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.95      0.97        20
       virginica       0.92      1.00      0.96        12
    
        accuracy                           0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    
    
